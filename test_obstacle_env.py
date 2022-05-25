import time
from ObstacleEnviroment.fetch.pick_dyn_lifted_obstacles import FetchPickDynLiftedObstaclesEnv,pretty_obs
from SubGoalEnv import SubGoalEnv

env = SubGoalEnv(env="obstacle_env", render_subactions=False)
obs = env.reset()

print("----------------------\nTest 5 random actions:\n----------------------")
for i in range(1):
    print("obs",pretty_obs(obs))
    env.render()
    goal_position = env.pretty_obs(obs)["goal"]
    action = env.scale_env_pos_to_action(goal_position)

    gripper_pos = env.pretty_obs(obs)['gripper_pos'][:3]
    gripper_pos[0] -= 0.1
    gripper_pos[1] += 0.1
    action = env.scale_env_pos_to_action(gripper_pos)
    action.append(1)
    # action = [-0.5931246039380769, 1.0,1,1]
    print("action:", action)
    print("env_pos for action:", env.scale_action_to_env_pos(action))
    obs, r, d, i = env.step(action)
    print("reward:",r)
    print("pretty obs:", pretty_obs(obs))
    print("info:",i,"\n")
while True:
    env.render()







