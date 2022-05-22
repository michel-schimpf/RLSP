# from gym.wrappers import Monitor
from typing import Tuple

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

    #create env
    mt10 = metaworld.MT10()
    env_array = []
    one_hot_index =0
    for name, env_cls in mt10.train_classes.items():
        for i in range(number_envs_of_each_task):
            env = SubGoalEnv(env=name,rew_type=rew_type,number_of_one_hot_tasks=10,one_hot_task_index=one_hot_index)
            env_array.append(lambda: env)
        one_hot_index +=1
    env_vec = SubprocVecEnv(env_array)
    env_vec = RLPPAMonitor(env_vec, "logs/PPO", ("success",))


    # create or load model
    model = PPO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=timestamps,
                 batch_size=batch_size,)
    # model = ALGO.load("models/PPO3/15360000.zip", env=env_vec,tensorboard_log=logdir)

    # safe models
    i = 0
    while True:
            print(i)
            i += 1
            model = model.learn(total_timesteps=timestamps, reset_num_timesteps=False,
                                tb_log_name="PPO", )
            model.save(f"{models_dir}/{timestamps * i * number_envs_of_each_task *10}")


if __name__ == '__main__':
    train()
