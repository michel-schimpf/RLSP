# Todo: let it inherit from env
import time
from typing import Tuple
import numpy as np
import gym
import metaworld
from gym.spaces import Box
from GripperControl import reach, pick, place


# Todo: add types

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


class NormalStepEnv(gym.Env):

    def render(self, mode="human"):
        self.env.render()

    def __init__(self, env="reach-v2", reward_type="", multi_task=0,):
        # set enviroment: todo: do it adjustable
        mt1 = metaworld.MT1(env)  # Construct the benchmark, sampling tasks
        env = mt1.train_classes[env]()  # Create an environment with task `pick_place`
        self.tasks = mt1.train_tasks
        self.cur_task_index = 0
        env.set_task(self.tasks[self.cur_task_index])  # Set task
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
        self._max_episode_length = 500
        self.number_steps = 0
        self.reward_type = reward_type

    def reset(self):
        self.cur_task_index += 1
        if self.cur_task_index >= len(self.tasks):
            self.cur_task_index = 0
        self.env.set_task(self.tasks[self.cur_task_index])
        obs = self.env.reset()
        self.number_steps = 0
        return new_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = -1
        if info["success"]:
            done = True
            reward = 1000
        self.number_steps += 1
        obs = new_obs(obs)
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info
