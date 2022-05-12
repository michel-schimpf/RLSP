from stable_baselines3 import TD3
import numpy as np
import os
from SubGoalEnv08042022 import SubGoalEnv
from stable_baselines3.common.noise import NormalActionNoise

models_dir = "models/TD3"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = SubGoalEnv(env="reach-v2",)
env.reset()

# The noise objects for TD3

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

TIMESTEPS = 2048
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")