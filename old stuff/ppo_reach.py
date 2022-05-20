#!/usr/bin/env python3
"""This is an example to train PPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import torch
from garage.envs import GymEnv
from garage import wrap_experiment
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from TestDUmmeyPixelEnv import PointEnv


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=3)
@click.option('--batch_size', default=100000)
@wrap_experiment(snapshot_mode='all')
def mtppo_metaworld_mt1_reach(ctxt, seed, epochs, batch_size):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.
    """
    # set_seed(seed)
    print("Start:")
    mt1 = metaworld.MT1('reach-v2')
    n_tasks = len(mt1.train_tasks)
    train_task_sampler = MetaWorldTaskSampler(mt1, 'train',
                                              lambda env, _: normalize(env))
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')
    print("Envs Rapped")
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    print("Sampler:")
    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)
    # load existing
    # snapshotter = Snapshotter()
    # snapshot = snapshotter.load('./data/local/experiment/mtppo_metaworld_mt1_reach')
    # algo = snapshot['algo']
    # #

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    print("Training:")
    trainer.train(n_epochs=epochs, batch_size=batch_size)


@click.command()
@wrap_experiment(snapshot_mode='all')
def try_subgoal_env(ctxt, epochs=3, batch_size=1000):
    print("Start:")
    env = GymEnv(PointEnv(), max_episode_length=100)

    print("Envs Rapped")
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    print("Sampler:")
    #Todo look in ray sampler
    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)
    # load existing
    # snapshotter = Snapshotter()
    # snapshot = snapshotter.load('./data/local/experiment/mtppo_metaworld_mt1_reach')
    # algo = snapshot['algo']
    # #

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    print("Training:")
    trainer.train(n_epochs=epochs, batch_size=batch_size)



from garage import wrap_experiment
from garage.envs import PointEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.trainer import TFTrainer


@wrap_experiment
def trpo_point(ctxt=None, seed=1):
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = normalize(PointEnv())

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000)