import gym
from gym.wrappers import Monitor
from stable_baselines3 import PPO
import os
import numpy as np
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor


def train():
    algo = "PPO"
    ALGO = PPO
    models_dir = f"models/{algo}"
    logdir = "logs"
    TIMESTEPS = 512
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
    env_vec = VecMonitor(env_vec, "logs/PPO_0")
    model = ALGO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=TIMESTEPS)
    # model = ALGO.load("models/PPO/3293184", env=env_vec)
    iters = 0
    while True:
        print(iters)
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=algo,)
        model.save(f"{models_dir}/{TIMESTEPS * iters*48}")


if __name__ == '__main__':
    train()


