import numpy as np

from SubGoalEnv import SubGoalEnv, scale_action_to_env_pos, scale_env_pos_to_action
from helper import pretty_obs_subgoal

env = SubGoalEnv("pick-place-v2", render_subactions=True)
obs = env.reset()
print("----------------------\nTest action to env point\n----------------------")
actions = [[-1, -1, -0.9, 0], [1, -1, -0.1, 0], [-1, 1, -0.1, 0], [1, 1, -0.1, 0],
           [-1, -1, 1, 0], [1, -1, 1, 0], [-1, 1, 1, 0], [1, 1, 1, 0], ]
total_reach = 0


# for a in actions:
#     obs = env.reset()
#     print("o:", obs[:4])
#     print("a:", a)
#     print("pos:", scale_action_to_env_pos(a))
#     obs, r, d, i1 = env.step(a)
#     # -----
#     print("---")
#     print("o:", obs[:4])
#     first_obj = pretty_obs_subgoal(obs)['first_obj']
#     print("first_obj:", first_obj)
#     action_to_reach_goal = scale_env_pos_to_action(first_obj)
#     action_to_reach_goal.append(1)
#     print("action:", action_to_reach_goal)
#     print("action in env pos:", scale_action_to_env_pos(action_to_reach_goal))
#     obs, r, d, i1 = env.step(action_to_reach_goal)
#     print("reward:", r)
#     print("info", i1)
#     #  -----
#     print("---")
#     print(pretty_obs_subgoal(obs))
#     goal = pretty_obs_subgoal(obs)['goal']
#     print("goal:", goal)
#     action_to_reach_goal = scale_env_pos_to_action(goal)
#     action_to_reach_goal.append(-1)
#     print("action:", action_to_reach_goal)
#     obs, r, d, i2 = env.step(action_to_reach_goal)
#     print("reward:", r)
#     print(i2)
#     if i2['success']:
#         print("reached with:", action_to_reach_goal)
#         total_reach += 1
#     else:
#         print("not reached with:", action_to_reach_goal)
#
#     print("--------------------------------------------")


for i in range(1000):
    obs = env.reset()
    print("o:", obs[:4])
    a = env.action_space.sample()
    print("a:", a)
    print("pos:", scale_action_to_env_pos(a))
    obs, r, d, i1 = env.step(a)
    print("o:", obs[:4])
    goal = pretty_obs_subgoal(obs)['first_obj']
    distance_to_subgoal = np.linalg.norm(obs[:3] - goal[:3])
    print("distance to object:", distance_to_subgoal)
    print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("reward:", r)
    print("info", i1)
    print(pretty_obs_subgoal(obs))
    goal = pretty_obs_subgoal(obs)['goal']
    print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-1)
    print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    print("reward:", r)
    print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
    else:
        print("not reached with:", action_to_reach_goal)

    print("--------------------------------------------")
print("total_rach of 8:", total_reach)



