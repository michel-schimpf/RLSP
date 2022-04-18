from SubGoalEnv import SubGoalEnv, scale_action_to_env_pos, new_obs


class SubGoalTrainEnv(SubGoalEnv):

    def step(self, action):
        obs = self.env.reset()
        # print("--obs:", obs)
        gripper_closed = True if action[3] > 0 else False #todo gripper ?
        # transform action into cordinates
        sub_goal_pos = scale_action_to_env_pos(action)
        # set gripper to subgoal position and get new obs
        # print("--subgoalpos:",[round(i, 7) for i in sub_goal_pos])
        self.env.hand_init_pos = [round(i, 7) for i in sub_goal_pos]
        self.env.reset_hand(1000)
        obs, reward, done, info = self.env.step([0,0,0,0])
        # print("--info:", info)
        # print("--obs:", obs)

        # get done and info

        # set reward
        reward = -1
        if info["success"]:
            reward = 100

        self.number_steps += 1
        if self.number_steps >= self._max_episode_length:
            info["TimeLimit.truncated"] = not done
            done = True
        obs = new_obs(obs)
        return obs, reward, done, info
