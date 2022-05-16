import time

import numpy as np

from SubGoalEnv_modular import SubGoalEnv, scale_action_to_env_pos, scale_env_pos_to_action, pretty_obs
env = SubGoalEnv("MT10", render_subactions=False)
obs = env.reset()
total_reach = 0
for i in range(50):
    # print("----------------------\nTest pick random actions:\n----------------------")
    # print(obs)
    goal = pretty_obs(obs)['first_obj']
    print("goal:", goal)
    action_to_reach_goal = env.action_space.sample()
    # print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    # print("reward:", r)
    # print("info", i1)
    print(env.cur_task_index)
    obs = env.reset()
