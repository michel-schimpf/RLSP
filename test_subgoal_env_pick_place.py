import time

import numpy as np

from SubGoalEnv import  SubGoalEnv, scale_action_to_env_pos, scale_env_pos_to_action
from helper import pretty_obs_subgoal
env = SubGoalEnv("pick-place-v2", render_subactions=True)
obs = env.reset()
total_reach =0
for i in range(3):
    # obs = env.reset()
    goal = pretty_obs_subgoal(obs)['first_obj']
    print("firsobj:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    print("correct action to reach obj", action_to_reach_goal)
    print("----------------------\nTest pick random actions:\n----------------------")
    print(obs)
    goal = pretty_obs_subgoal(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = [ 0.32396266, -1.    ,     -0.9    ,     -0.557377  ]
    # action_to_reach_goal.append(1)
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info",i1)
    goal = pretty_obs_subgoal(obs)['first_obj']
    distance_to_subgoal = np.linalg.norm(obs[:3] - goal[:3])
    print("distance to object:", distance_to_subgoal)
    print("----------------------\nTest pick random actions:\n----------------------")
    print(obs)
    goal = pretty_obs_subgoal(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = [0.32396266, -0.6, -0.9, -0.557377]
    # action_to_reach_goal.append(1)
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info", i1)
    goal = pretty_obs_subgoal(obs)['first_obj']
    distance_to_subgoal = np.linalg.norm(obs[:3] - goal[:3])
    print("distance to object:", distance_to_subgoal)

    # print("----------------------\nTest pick random actions:\n----------------------")
    # print(obs)
    # goal = pretty_obs_subgoal(obs)['first_obj']
    # print("goal:", goal)
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(1)
    # # action_to_reach_goal[0] += 0.5
    # print("action:", action_to_reach_goal)
    # obs, r, d, i1 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print("info", i1)
    # print("----------------------\nTest pick random actions:\n----------------------")
    # print(obs)
    # goal = pretty_obs_subgoal(obs)['first_obj']
    # print("goal:", goal)
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(0.1)
    # # action_to_reach_goal[1] += 0.5
    # print("action:", action_to_reach_goal)
    # obs, r, d, i1 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print("info", i1)
    # print("----------------------\nTest pick random actions:\n----------------------")
    # print(obs)
    # goal = pretty_obs_subgoal(obs)['first_obj']
    # print("goal:", goal)
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(0.1)
    # # action_to_reach_goal[0] -= 0.1
    # print("action:", action_to_reach_goal)
    # obs, r, d, i1 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print("info", i1)
    # print("----------------------\nTest drop random actions:\n----------------------")
    # print(pretty_obs_subgoal(obs))
    # goal = pretty_obs_subgoal(obs)['goal']
    # print("goal:", goal)
    # goal[1] -= 0.5
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(-0.1)
    # print("action:", action_to_reach_goal)
    # obs, r, d, i2 = env.step(action_to_reach_goal)
    # print("reward:",r)
    # print(i2)
    # if i2['success']:
    #     print("reached with:",action_to_reach_goal)
    #     total_reach +=1
    # else:
    #     print("not reached with:",action_to_reach_goal)
    # print("----------------------\nTest drop random actions:\n----------------------")
    # print(pretty_obs_subgoal(obs))
    # goal = pretty_obs_subgoal(obs)['goal']
    # print("goal:", goal)
    # goal[0] -= 0.5
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(-0.1)
    # print("action:", action_to_reach_goal)
    # obs, r, d, i2 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print(i2)
    # if i2['success']:
    #     print("reached with:", action_to_reach_goal)
    #     total_reach += 1
    # else:
    #     print("not reached with:", action_to_reach_goal)
    # print("----------------------\nTest drop random actions:\n----------------------")
    # print(pretty_obs_subgoal(obs))
    # goal = pretty_obs_subgoal(obs)['goal']
    # print("goal:", goal)
    # goal[0] -= 0.1
    # goal[1] -= 0.1
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(-1)
    # print("action:", action_to_reach_goal)
    # obs, r, d, i2 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print(i2)
    # if i2['success']:
    #     print("reached with:", action_to_reach_goal)
    #     total_reach += 1
    # else:
    #     print("not reached with:", action_to_reach_goal)
    # print("----------------------\nTest drop random actions:\n----------------------")
    # print(pretty_obs_subgoal(obs))
    # goal = pretty_obs_subgoal(obs)['goal']
    # print("goal:", goal)
    # # goal[0] = 0.1
    # action_to_reach_goal = scale_env_pos_to_action(goal)
    # action_to_reach_goal.append(-0.1)
    # print("action:", action_to_reach_goal)
    # obs, r, d, i2 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print(i2)
    # if i2['success']:
    #     print("reached with:", action_to_reach_goal)
    #     total_reach += 1
    # else:
    #     print("not reached with:", action_to_reach_goal)
    #
    # print()
print(f"\n\n--------------- \nreached:{total_reach}")