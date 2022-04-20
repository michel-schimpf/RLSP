from SubGoalEnv import SubGoalEnv, scale_action_to_env_pos, scale_env_pos_to_action
from helper import pretty_obs_subgoal

env = SubGoalEnv("pick-place-v2", render_subactions=False)
obs = env.reset()
print("----------------------\nTest action to env point\n----------------------")
actions = [[-1, -1, -0.9, 0], [1, -1, -0.1, 0], [-1, 1, -0.1, 0], [1, 1, -0.1, 0],
           [-1, -1, 1, 0], [1, -1, 1, 0], [-1, 1, 1, 0], [1, 1, 1, 0], ]
for a in actions:
    print("o:", obs[:4])
    print("a:", a)
    print("pos:", scale_action_to_env_pos(a))
    obs, r, d, i1 = env.step(a)
    print("o:", obs[:4])
    print("--------------------------------------------")



print("----------------------\nTest action to env point\n----------------------")
for a in actions:
    obs = env.reset()
    print("o:", obs[:4])
    print("a:", a)
    print("pos:", scale_action_to_env_pos(a))
    obs, r, d, i1 = env.step(a)
    print("o:", obs[:4])
    print("--------------------------------------------")
