import time
from helper import pretty_obs_subgoal
import gym
from stable_baselines3 import PPO,DDPG,SAC
from SubGoalEnv import SubGoalEnv, scale_action_to_env_pos
ALGO = PPO

models_dir = "models/PPO"

env = SubGoalEnv("pick-place-v2", render_subactions=True)
# env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
env.reset()

model_path = f"{models_dir}/2531328.zip"
model = ALGO.load(model_path, env=env)
episodes = 50
mean_rew_all_tasks =0
mean_steps= 0
for ep in range(episodes):
    print("\n---------\nepisode:", ep)
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    while not done:
        action, _states = model.predict(obs)
        print("obs:", pretty_obs_subgoal(obs))
        print("action:", action)
        print("intended subgoal:", scale_action_to_env_pos(action))
        obs, reward, done, info = env.step(action)
        print("obs after action:",pretty_obs_subgoal(obs))
        print(info)
        print("reward:", reward)
        steps += 1
        total_reward += reward
        print()
    print("total reward:",total_reward)
    print("mean reward:",total_reward/steps)
    print("finished after: ", steps, " steps \n")
    mean_rew_all_tasks += total_reward
    mean_steps += steps
print("mean_tot_rew:",mean_rew_all_tasks/50)
print("mean_steps:", mean_steps/50)

