# from helper import pretty_obs_subgoal
from stable_baselines3 import PPO

from SubGoalEnv import SubGoalEnv, pretty_obs,scale_action_to_env_pos
ALGO = PPO

models_dir = "models/cloud_models"

env = SubGoalEnv("pick-place-v2", render_subactions=True)

model_path = f"{models_dir}/10158080"
model = ALGO.load(model_path, env=env)
episodes = 100
mean_rew_all_tasks = 0
num_success = 0
mean_steps = 0
for ep in range(episodes):
    print("\n---------\nepisode:", ep)
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    success = False
    while not done:

        action, _states = model.predict(obs, deterministic=True)
        print("action:", action)
        print("intended subgoal:", scale_action_to_env_pos(action))
        obs, reward, done, info = env.step(action)
        # print("gripper pos",pretty_obs(obs)["gripper"])
        print("goal:", pretty_obs(obs)["goal"])
        obj = pretty_obs(obs)['first_obj']
        # distance_to_subgoal = np.linalg.norm(obs[:3] - obj[:3])
        # print("distance to object:", distance_to_subgoal)
        print("info",info)
        print("reward:", reward)
        steps += 1
        total_reward += reward
        if info['success']:
            success = True
    #     print()
        if done and success:
            num_success += 1
    print("total reward:",total_reward)
    # print("mean reward:",total_reward/steps)
    print("finished after: ", steps, " steps \n")
    mean_rew_all_tasks += total_reward
    mean_steps += steps
print("mean_tot_rew:",mean_rew_all_tasks/episodes)
print("mean_steps:", mean_steps/episodes)
print("success rate:",num_success/episodes)

