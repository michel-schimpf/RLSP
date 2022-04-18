# Todo: let it inherit from env
from typing import Tuple
import numpy as np
import gym
import metaworld
from gym.spaces import Box
from GripperControl import reach, pick, place


# Todo: add types


# not tested jet
def scale_action_to_env_pos(action):
    action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.04604, 0.49672)]  # figured out by trying
    env_pos = []
    for i in range(3):
        action_range = (action_dimension[i][1] - action_dimension[i][0])
        env_range = (env_dimension[i][1] - env_dimension[i][0])
        env_pos.append((((action[i] - action_dimension[i][0]) * env_range) / action_range) + env_dimension[i][0])
    return env_pos


def scale_env_pos_to_action(env_pos):
    action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.04604, 0.49672)]  # figured out by trying
    action = []
    for i in range(3):
        action_range = (action_dimension[i][1] - action_dimension[i][0])
        env_range = (env_dimension[i][1] - env_dimension[i][0])
        action.append((((env_pos[i] - env_dimension[i][0]) * action_range) / env_range) + action_dimension[i][0])
    return action


def pretty_obs(obs):
    return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
            'goal': obs[36:39], }  # 'last_measurements': obs[18:36]}


def new_obs(obs):
    po = pretty_obs(obs)
    x = po['gripper_pos']
    x = np.append(x, po['first_obj'])
    x = np.append(x, po['second_obj'])
    x = np.append(x, po['goal'])
    return x


# Todo: maybe take gym.env?
class SubGoalEnv(gym.Env):

    def render(self, mode="human"):
        self.env.render()

    def __init__(self,env="reach-v2", reward_type="", task=0, render_subactions=False):
        # set enviroment: todo: do it adjustable
        mt1 = metaworld.MT1(env)  # Construct the benchmark, sampling tasks
        env = mt1.train_classes[env]()  # Create an environment with task `pick_place`
        self.task = mt1.train_tasks[task]  # todo: check if works
        env.set_task(self.task)  # Set task
        self.env = env

        # define action space:
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]), dtype=np.float32
        )

        # define oberservation space (copied from sawyer_xyz_env
        hand_space = Box(
            np.array([-0.525, .348, -.0525]),
            np.array([+0.525, 1.025, .7]), dtype=np.float32
        )
        gripper_low = -1.
        gripper_high = +1.
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3)
        goal_high = np.zeros(3)
        self.observation_space = Box(
            np.hstack((hand_space.low, gripper_low, obj_low, goal_low)),
            np.hstack((hand_space.high, gripper_high, obj_high, goal_high)), dtype=np.float32
        )
        # other
        self._max_episode_length = 20
        self.number_steps = 0
        self.reward_type = reward_type
        self.render_subactions = render_subactions


    def reset(self):
        obs = self.env.reset()
        self.number_steps =0
        return new_obs(obs)

    def step(self, action):
        obs = self.env.reset()
        gripper_closed = True if action[3] > 0 else False
        # transform action into cordinates
        sub_goal_pos = scale_action_to_env_pos(action)

        # find trajectory to reach coordinates
        # use tcp_center because its more accurat then obs
        sub_actions = reach(current_pos=self.env.tcp_center, goal_pos=sub_goal_pos, gripper_closed=gripper_closed)

        # loop over actions and perform actions on enviroment
        done = False
        info = {}
        reward= 0
        for a in sub_actions:
            obs, reward, done, info = self.env.step(a)
            if self.render_subactions:
                self.env.render()
            # print(pretty_obs(obs))
            done = info['success']
        # if it did not reach completly do again
        distance_to_subgoal= np.linalg.norm(self.env.tcp_center - sub_goal_pos)
        # print("--distance_to_subgoal: ", distance_to_subgoal)
        if distance_to_subgoal > 0.05:
            # print("--distance near_object is greater 0.05: ", np.linalg.norm(obs[0:3] - sub_goal_pos))
            sub_actions = reach(current_pos=self.env.tcp_center, goal_pos=sub_goal_pos, gripper_closed=gripper_closed)
            # print("--subactions:",sub_actions)
            for a in sub_actions:
                obs, reward, done, info = self.env.step(a)
                if self.render_subactions:
                    self.env.render()
                # print(pretty_obs(obs))
                done = info['success']
            # print("--new distance: ", np.linalg.norm(self.env.tcp_center - sub_goal_pos))
        reward = -1
        if done:
            reward = 100
        # calculate reward:
        if self.reward_type == 'sparse':
            reward = 1 if info['success'] else 0

        self.number_steps += 1
        obs = new_obs(obs)
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info
