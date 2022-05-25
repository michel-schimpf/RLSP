# from gym.wrappers import Monitor
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO

import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from RL_PPA_monitor import RLPPAMonitor


def train():
    # variables:
    models_dir = f"models/PPO"
    logdir = "logs"
    timestamps = 2048
    number_envs_of_each_task = 6
    batch_size = 4096
    rew_type = "rew1"

    # create env
    mt10 = metaworld.MT10()
    env_array = []
    for i in range(number_envs_of_each_task):
        env_array += [make_env(name, rew_type, 10, i) for i, (name, _) in enumerate(mt10.train_classes.items())]
    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec, "logs/PPO", multi_env=True)

    # create or load model
    model = PPO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=timestamps,
                batch_size=batch_size, )
    # model = ALGO.load("models/PPO3/15360000.zip", env=env_vec,tensorboard_log=logdir)

    # safe models
    i = 0
    while True:
        print(i)
        i += 1
        model = model.learn(total_timesteps=timestamps, reset_num_timesteps=False,
                            tb_log_name="PPO", )
        model.save(f"{models_dir}/{timestamps * i * number_envs_of_each_task * 10}")


def make_env(name,rew_type,number_of_one_hot_tasks,one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                      one_hot_task_index=one_hot_task_index)
    return _init


if __name__ == '__main__':
    train()
