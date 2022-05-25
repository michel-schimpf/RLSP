# Todo: let it inherit from env
import enum
import time
from typing import Tuple, Dict
import numpy as np
import gym
import metaworld
from gym.spaces import Box
from GripperControl import reach, Obstacles
from metaworld.envs import reward_utils
from ObstacleEnviroment.fetch.pick_dyn_lifted_obstacles import FetchPickDynLiftedObstaclesEnv, pretty_obs

# ENV_DIMENSION = [(-0.37, 0.31), (0.40, 0.91), (0.0, 0.31)]
# for pick_place: minimum = [(-0.15, 0.15), (0.58, 0.91), (0.0, 0.31)]
# with peg_insert: [(-0.36, 0.26), (0.39, 0.91), (0.0, 0.31)]
# with door close [(-0.37, 0.31), (0.40, 0.91), (0.0, 0.31)]


class SubGoalEnv(gym.Env):
    # todo: order methods
    def __init__(self, env="reach-v2", render_subactions=False, rew_type="",
                 number_of_one_hot_tasks=1, one_hot_task_index=-1):
        rew_types = ["","meta_world_rew","rew1"]
        if rew_type not in rew_types:
            raise Exception('rew_type needs to be one of: ', rew_types)
        self.env_rew = rew_type
        self.env_name = env
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]), dtype=np.float32
        )
        self._max_episode_length = 20
        self.number_steps = 0
        self.render_subactions = render_subactions
        self.already_grasped = False
        self.number_of_one_hot_tasks = number_of_one_hot_tasks
        self.one_hot_task_index = one_hot_task_index
        # different for each env
        if self.env_name == "obstacle_env":
            # set enviroment dimensions
            # todo: check
            self.env_dimension = [(1.15, 1.45), (0.65, 1.1), (0.4, 0.5)]
            # set enviorment
            self.env = FetchPickDynLiftedObstaclesEnv()
            # set observation space
            obs = self.env.reset()['observation']
            self.observation_space = Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")
        else:
            # set enviroment dimensions
            self.env_dimension = [(-0.37, 0.31), (0.40, 0.91), (0.0, 0.31)]
            # set enviroment
            mt1 = metaworld.MT1(env)  # Construct the benchmark, sampling tasks
            env = mt1.train_classes[env]()  # Create an environment with task `pick_place`
            self.tasks = mt1.train_tasks
            self.cur_task_index = 0
            env.set_task(self.tasks[self.cur_task_index])  # Set task
            self.env = env

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
            # different for multi task and non multi task env
            if number_of_one_hot_tasks ==1:
                self.observation_space = Box(
                    np.hstack((hand_space.low, gripper_low, obj_low,
                               hand_space.low, gripper_low, obj_low, goal_low)),
                    np.hstack((hand_space.high, gripper_high, obj_high,
                               hand_space.high, gripper_high, obj_high, goal_high))
                    , dtype=np.float32)
            else:
                one_hot = Box(np.ones(10), np.ones(10)*-1, dtype=np.float32)
                self.observation_space = Box(
                    np.hstack((hand_space.low, gripper_low, obj_low,
                               hand_space.low, gripper_low, obj_low, goal_low,one_hot.low)),
                    np.hstack((hand_space.high, gripper_high, obj_high,
                               hand_space.high, gripper_high, obj_high, goal_high,one_hot.high))
                    , dtype=np.float32)

    def pretty_obs(self, obs):
        if self.env_name == "obstacle_env":
            return pretty_obs(obs)
        return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
                'goal': obs[36:39], 'last_measurements': obs[18:36], "one_hot_task": obs[39:]}

    def scale_action_to_env_pos(self,action):
        action = np.clip(action, -1, 1)
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        env_pos = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            env_pos.append((((action[i] - action_dimension[i][0]) * env_range) / action_range) + self.env_dimension[i][0])
        return env_pos

    def scale_env_pos_to_action(self,env_pos):
        action_dimension = [(-1, 1), (-1, 1), (-1, 1)]
        action = []
        for i in range(3):
            action_range = (action_dimension[i][1] - action_dimension[i][0])
            env_range = (self.env_dimension[i][1] - self.env_dimension[i][0])
            action.append((((env_pos[i] - self.env_dimension[i][0]) * action_range) / env_range) + action_dimension[i][0])
        action = list(np.clip(action, -1, 1))
        return action

    def _change_obs(self, obs) ->[float]:
        # if obstacle env change obs
        if self.env_name == "obstacle_env":
            return obs["observation"]
        # if normal env just return obs
        if self.number_of_one_hot_tasks <= 1:
            return obs
        # if multi env add one-hot
        one_hot = np.zeros(self.number_of_one_hot_tasks)
        one_hot[self.one_hot_task_index] = 1
        return np.concatenate([obs, one_hot])

    def _calculate_reward(self, re, info: Dict[str, bool], obs: [float], actiontype) -> (int, bool):
        # todo: refactor
        reward = -2
        done = False
        if self.env_rew in ["meta_world_rew","normal",""]:
            return re, done
        if self.env_rew == "rew1":
            if info['success']:
                return 0, True
            return (-1 + (re/10)), False
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
            gripper_to_obj_reward = reward_utils.tolerance(gripper_to_obj,bounds=(0, _TARGET_RADIUS),
                                                           margin=in_place_margin,sigmoid='long_tail', )
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
            # Todo: check if neccessary with already grasped
            obj_to_goal_reward = 0

            if is_grasped and not (self.already_grasped and actiontype == 1) and 'in_place_reward' in info:
                obj_to_goal_reward = info['in_place_reward']
            # return total reward
            # print("original reward:", re)

            if 'success' in info and info['success']:
                return 0, True
            else:
                # print(f"reward compontents: g_to_obj_r: {gripper_to_obj_reward}, grasp_r: {grasp_reward}, obj_to_g_r: {obj_to_goal_reward}")
                return (
                                   reward + gripper_to_obj_reward * 1 / 6 + grasp_reward * 2 / 6 + obj_to_goal_reward * 3 / 6), False

    def render(self, mode="human"):
        self.env.render()

    def reset(self):
        if self.env_name != "obstacle_env":
            if self.cur_task_index >= len(self.tasks):
                self.cur_task_index = 0
            self.env.set_task(self.tasks[self.cur_task_index])
            self.cur_task_index += 1
            self.already_grasped = False
        self.number_steps = 0
        obs = self.env.reset()
        return self._change_obs(obs)

    def step(self, action):
        # obs = [0] * 40
        # get kind of action: "hold"=0, "grap"=1
        actiontype = 0
        gripper_closed = True
        if action[3] > 0:
            actiontype = 1
            gripper_closed = False

        # transform action into cordinates
        sub_goal_pos = self.scale_action_to_env_pos(action)

        #create inital obs,
        obs, reward, done, info = self.env.step([0, 0, 0, 0])

        if actiontype == 1:
            # open gripper if picking
            for i in range(15):
                obs, reward, done, info = self.env.step([0, 0, 0, -1])
                if self.render_subactions:
                    self.env.render()
                    time.sleep(0.05)

        if self.env_name == "obstacle_env":
            gripper_pos = obs["observation"][:3]
            obstacles = Obstacles(pretty_obs(obs["observation"]), self.env.dt)
            # print(obstacles)
        else:
            obstacles = None
            gripper_pos = self.env.tcp_center

        max_it = 3
        if self.env_name == "obstacle_env":
            max_it = 1

        # if it did not reach completly do again
        while np.linalg.norm(gripper_pos - sub_goal_pos) > 0.0005:
            if self.env_name == "obstacle_env":
                gripper_pos = obs["observation"][:3]
            else:
                gripper_pos = self.env.tcp_center

            sub_actions = reach(current_pos=gripper_pos, goal_pos=sub_goal_pos,
                                gripper_closed=gripper_closed, obstacles=obstacles, env_dimension=self.env_dimension)
            if sub_actions is None:
                sub_actions = [[0,0,0,0]]
            for a in sub_actions:
                # print("bevor",pretty_obs(obs['observation']))
                obs, reward, done, info = self.env.step(a)
                # print("--sa:", a)
                # print("after", pretty_obs(obs['observation']))
                if self.render_subactions:

                    self.env.render()
                    time.sleep(0.05)
            max_it -= 1
            if max_it == 0:
                break
        # do picking or droping depending on action type:
        if actiontype == 1:
            for i in range(15):
                obs, reward, done, info = self.env.step([0, 0, 0, 1])
                if self.render_subactions:
                    self.env.render()
                    time.sleep(0.05)
        # calculate reward
        reward, done = self._calculate_reward(reward, info, obs, actiontype)
        self.number_steps += 1
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        # TODO: get real reward
        return self._change_obs(obs), reward, done, info
