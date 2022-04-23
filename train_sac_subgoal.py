import gym
from gym.wrappers import Monitor
from stable_baselines3 import SAC
import os
import numpy as np
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor


def train():
    algo = "SAC"
    ALGO = SAC
    models_dir = f"models/{algo}"
    logdir = "logs"
    TIMESTEPS = 5000

    env = SubGoalEnv("pick-place-v2")
    env_vec = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             ])
    env_vec = VecMonitor(env_vec, "logs/SAC_0")
    model = ALGO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir,)
    # model = ALGO.load("models/SAC/1220000", env=env_vec)
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=algo,)
        model.save(f"{models_dir}/{TIMESTEPS * iters}")


if __name__ == '__main__':
    train()


