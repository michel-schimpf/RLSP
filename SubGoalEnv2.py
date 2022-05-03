# Todo: let it inherit from env
import enum
import time
from typing import Tuple, Dict
import numpy as np
import gym
import metaworld
from gym.spaces import Box
from GripperControl import reach, pick, place
from metaworld.envs import reward_utils


# Todo: add types


def scale_action_to_env_pos(action):
    action = np.clip(action, -1, 1)
    action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.04604, 0.49672)]  # figured out by trying
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.0, 0.49672)]  # add a bit of marging
    env_pos = []
    for i in range(3):
        action_range = (action_dimension[i][1] - action_dimension[i][0])
        env_range = (env_dimension[i][1] - env_dimension[i][0])
        env_pos.append((((action[i] - action_dimension[i][0]) * env_range) / action_range) + env_dimension[i][0])
    return env_pos


def scale_env_pos_to_action(env_pos):
    action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.04604, 0.49672)]  # figured out by trying
    env_dimension = [(-0.50118, 0.50118), (0.40008, 0.9227), (0.0, 0.49672)]  # add a bit of marging
    action = []
    for i in range(3):
        action_range = (action_dimension[i][1] - action_dimension[i][0])
        env_range = (env_dimension[i][1] - env_dimension[i][0])
        action.append((((env_pos[i] - env_dimension[i][0]) * action_range) / env_range) + action_dimension[i][0])
    action = list(np.clip(action, -1, 1))
    return action


def pretty_obs(obs):
    return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
            'goal': obs[36:39], 'last_measurements': obs[18:36]}

#
# def new_obs(obs):
#     po = pretty_obs(obs)
#     x = po['gripper_pos']
#     x = np.append(x, po['first_obj'])
#     x = np.append(x, po['second_obj'])
#     x = np.append(x, po['goal'])
#     return x


# Todo: maybe take gym.env?
class SubGoalEnv(gym.Env):

    def __init__(self, env="reach-v2", multi_task=0, render_subactions=False):
        # set enviroment: todo: do it adjustable
        self.env_name = env
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
            np.hstack((hand_space.low, gripper_low, obj_low,
                       hand_space.low, gripper_low, obj_low, goal_low)),
            np.hstack((hand_space.high, gripper_high, obj_high,
                       hand_space.high, gripper_high, obj_high, goal_high))
            , dtype=np.float32
        )
        # other
        self._max_episode_length = 20
        self.number_steps = 0
        self.render_subactions = render_subactions
        self.already_grasped = False

    def _calculate_reward(self, re, info: Dict[str, bool], obs: [float], actiontype ) -> (int, bool):
        reward = -1
        done = False
        if self.env_name == "reach-v2":
            reward = -1
            if 'success' in info and info['success']:
                reward = 10
                done = True
        elif self.env_name == "pick-place-v2":
            # give reward for distance to object
            _TARGET_RADIUS = 0.03
            obj_pos = pretty_obs(obs)['first_obj'][:3]
            gripper_pos = self.env.tcp_center
            gripper_to_obj = np.linalg.norm(obj_pos - gripper_pos)
            in_place_margin = (np.linalg.norm(self.env.hand_init_pos - obj_pos))
            gripper_to_obj_reward = reward_utils.tolerance(gripper_to_obj,
                                              bounds=(0, _TARGET_RADIUS),
                                              margin=in_place_margin,
                                              sigmoid='long_tail', )

            # give reward for grasping the object
            grasp_reward = 0
            if 'grasp_reward' in info:
                grasp_reward = info['grasp_reward']

            # if already grasped and grasped again, give negativ reward
            is_grasped = grasp_reward > 0.42
            if is_grasped:
                if self.already_grasped and actiontype == 1:
                    grasp_reward = 0
            else:
                if actiontype == 0:
                    grasp_reward = 0
            self.already_grasped = is_grasped
            if info['grasp_success']:
                grasp_reward = 1

            # if grasped give reward for how near the object is to goal position
            #Todo: check if neccessary with already grasped
            obj_to_goal_reward = 0

            if is_grasped and not(self.already_grasped and actiontype == 1) and 'in_place_reward' in info:
                obj_to_goal_reward = info['in_place_reward']
            # return total reward
            # print("original reward:", re)
            if 'success' in info and info['success']:
                return 0, True
            else:
                # print(f"reward compontents: g_to_obj_r: {gripper_to_obj_reward}, grasp_r: {grasp_reward}, obj_to_g_r: {obj_to_goal_reward}")
                return (reward + gripper_to_obj_reward * 1/6 + grasp_reward * 2/6 + obj_to_goal_reward * 3/6), False

    def render(self, mode="human"):
        self.env.render()

    def reset(self):
        if self.cur_task_index >= len(self.tasks):
            self.cur_task_index = 0
        self.env.set_task(self.tasks[self.cur_task_index])
        obs = self.env.reset()
        self.number_steps = 0
        self.cur_task_index += 1
        self.already_grasped = False
        return obs

    def step(self, action):
        obs = [0] * 40
        # get kind of action: "hold"=0, "grap"=1
        actiontype = 0
        gripper_closed = True
        if action[3] > 0:
            actiontype = 1
            gripper_closed = False

        # transform action into cordinates
        sub_goal_pos = scale_action_to_env_pos(action)

        # first reach above object so gripper does not collide
        if actiontype == 1:
            sub_goal_pos[2] += 0.04

        # find trajectory to reach coordinates
        # use tcp_center because its more accurat then obs
        sub_actions = reach(current_pos=self.env.tcp_center, goal_pos=sub_goal_pos, gripper_closed=gripper_closed)
        reward = 0
        if len(sub_actions) == 0:
            obs, reward, done, info = self.env.step([0, 0, 0, 0])
            reward, done = self._calculate_reward(reward, info, obs, actiontype)
            self.number_steps += 1
            if self.number_steps >= self._max_episode_length:
                info["TimeLimit.truncated"] = not done
                done = True
            return obs, reward, done, info

        # open gripper if picking
        if actiontype == 1:
            for i in range(4):
                obs, reward, done, info = self.env.step(place())
                if self.render_subactions:
                    # print("render")
                    self.env.render()
                    time.sleep(0.05)

        # loop over actions and perform actions on enviroment
        info = {}
        for a in sub_actions:
            obs, reward, done, info = self.env.step(a)
            if self.render_subactions:
                self.env.render()
                time.sleep(0.05)

        # if it did not reach completly do again
        distance_to_subgoal = np.linalg.norm(self.env.tcp_center - sub_goal_pos)
        if distance_to_subgoal > 0.01:
            sub_actions = reach(current_pos=self.env.tcp_center, goal_pos=sub_goal_pos, gripper_closed=gripper_closed)
            for a in sub_actions:
                obs, reward, done, info = self.env.step(a)
                if self.render_subactions:
                    # print("---i:",info)
                    self.env.render()
                    time.sleep(0.05)
        # do picking or droping depending on action type:
        # distance_to_subgoal = np.linalg.norm(self.env.tcp_center - sub_goal_pos)
        # print("--distance to subgoal:", distance_to_subgoal)
        if actiontype == 1:
            # # do one action to go lowe
            max_it = 5
            sub_goal_pos[2] -= 0.045
            while np.linalg.norm(self.env.tcp_center - sub_goal_pos) > 0.0005:
                sub_actions = reach(current_pos=self.env.tcp_center, goal_pos=sub_goal_pos,
                                    gripper_closed=gripper_closed)
                for a in sub_actions:
                    obs, reward, done, info = self.env.step(a)
                    if self.render_subactions:
                        # print("---i:",info)
                        self.env.render()
                        time.sleep(0.05)
                max_it -= 1
                if max_it == 0:
                    break
            for i in range(7):
                obs, reward, done, info = self.env.step([0,0, 0,1])
                if self.render_subactions:
                    # print("render")
                    self.env.render()
                    time.sleep(0.05)
        # calculate reward
        reward, done = self._calculate_reward(reward, info, obs, actiontype)
        self.number_steps += 1
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info
