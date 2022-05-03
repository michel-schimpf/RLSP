import time

import numpy as np

from SubGoalEnv2 import SubGoalEnv, scale_action_to_env_pos, scale_env_pos_to_action, pretty_obs
env = SubGoalEnv("pick-place-v2", render_subactions=False)
obs = env.reset()
total_reach = 0
for i in range(100):
    obs = env.reset()
    print("----------------------\nTest pick random actions:\n----------------------")
    print(obs)
    goal = pretty_obs(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = env.action_space.sample()
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info", i1)
    print("----------------------\nTest pick random actions:\n----------------------")
    print(obs)
    goal = pretty_obs(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    action_to_reach_goal[0] += 0.1
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info", i1)
    print("----------------------\nTest pick random actions:\n----------------------")
    print(obs)
    goal = pretty_obs(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info", i1)
    print("----------------------\nTest drop random actions:\n----------------------")
    print(pretty_obs(obs))
    goal = pretty_obs(obs)['goal']
    print("goal:", goal)

    action_to_reach_goal = [ 1.       ,   0.37243828, -0.8     ,    -0.66311646]#scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(-0.1)
    print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    print("reward:", r)
    print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
        continue
    else:
        print("not reached with:", action_to_reach_goal)
    print("----------------------\nTest drop random actions:\n----------------------")
    print(pretty_obs(obs))
    goal = pretty_obs(obs)['goal']
    print("goal:", goal)
    goal[0] -= 0.5
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-0.1)
    print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    print("reward:", r)
    print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
        continue
    else:
        print("not reached with:", action_to_reach_goal)
    print("----------------------\nTest drop random actions:\n----------------------")
    print(pretty_obs(obs))
    goal = pretty_obs(obs)['goal']
    print("goal:", goal)
    goal[0] -= 0.1
    goal[1] -= 0.1
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-1)
    print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    print("reward:", r)
    print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
        continue
    else:
        print("not reached with:", action_to_reach_goal)
    print("----------------------\nTest drop random actions:\n----------------------")
    print(pretty_obs(obs))
    goal = pretty_obs(obs)['goal']
    print("goal:", goal)
    # goal[0] = 0.1
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-0.1)
    print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    print("reward:", r)
    print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
        continue
    else:
        print("not reached with:", action_to_reach_goal)

    print()
print(f"\n\n--------------- \nreached:{total_reach}")