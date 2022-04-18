# from stable_baselines3.common.callbacks import EvalCallback
# from ppo_reach import mtppo_metaworld_mt1_reach, try_subgoal_env, trpo_point
from SubGoalEnv import SubGoalEnv

from stable_baselines3 import PPO


def pretty_obs(obs):
    return {'gripper_pos': obs[0:4], 'first_obj': obs[4:11], 'second_obj': obs[11:18],
            'goal': obs[36:39], }  # 'last_measurements': obs[18:36]}


def obs_diff(obs1, obs2):
    diff = []
    for i, _ in enumerate(obs1):
        diff.append(obs1[i] - obs2[i])
    return [round(i, 5) for i in diff]

#
# # Todo: try to put it in reach
# def get_last_action(cur_pos: [float], goal_pos: [float], step_size: float) -> [float]:
#     # dif = [x1 - x2 for (x1, x2) in zip(cur_pos, goal_pos)]
#     # action = [x / step_size for x in dif]
#     # action.append(1)
#     # return action
#     pass
#
#
# def main():
#     # ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']
#     mt1 = metaworld.MT1('pick-place-v2')  # Construct the benchmark, sampling tasks
#     env = mt1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
#     task = random.choice(mt1.train_tasks)
#     env.set_task(task)  # Set task
#     obs = env.reset()  # Reset environment
#     obs_pretty = pretty_obs(obs)
#
#
#     # get actions
#     actions = reach(current_pos=obs_pretty['gripper_pos'], goal_pos=obs_pretty['first_obj'], gripper_closed=False)
#     print(actions)
#     obs1 = obs
#     while actions:
#         env.render()
#         a = actions.pop(0)  # env.action_space.sample()
#         obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action\
#         print(a, reward)
#         print(pretty_obs(obs))
#         obs1 = obs
#
#     obs, reward, done, info = env.step(pick())
#     print(pick(), )
#     print(obs)
#     obs, reward, done, info = env.step(pick())
#     print(pick())
#     print(obs)
#     # reach:
#     print("----------------------")
#     obs_pretty = pretty_obs(obs)
#     actions = reach(current_pos=obs_pretty['gripper_pos'], goal_pos=obs_pretty['goal'], gripper_closed=True)
#     print(actions)
#     obs1 = obs
#     while actions:
#         env.render()
#         a = actions.pop(0)  # env.action_space.sample()
#         obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action\
#         print(info)
#         print(pretty_obs(obs))
#         print(obs_diff(obs, obs1))
#         obs1 = obs
#     for i in range(3000):
#         env.render()
#
#
# def ppo_reach_test():
#     from garage.experiment import Snapshotter
#     import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
#     snapshotter = Snapshotter()
#     with tf.compat.v1.Session(): # optional, only for TensorFlow
#         data = snapshotter.load('./data/local/experiment/mtppo_metaworld_mt1_reach_2')
#     policy = data['algo'].policy
#     # create
#     mt1 = metaworld.MT1('reach-v2')
#     env = mt1.train_classes['reach-v2']()
#     task = random.choice(mt1.train_tasks)
#     env.set_task(task)
#     steps, max_steps = 0, 5000
#     done = False
#     obs = env.reset()  # The initial observation
#     print(obs)
#     policy.reset()
#     while steps < max_steps and not done:
#         action = policy.get_action(np.array(obs))
#         obs, rewards, done, info = env.step(action[0])
#         env.render()  # Render the environment to see what's going on (optional)
#         print(info)
#         if info['success']:
#             print(steps)
#             return
#         steps += 1
#
#     env.close()


def subgoal_env_policy():
    # Parallel environments
    env = SubGoalEnv() #make_vec_env("CartPole-v1", n_envs=4)
    # env = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env])
    # # check_env(env)
    #
    # # model = PPO("MlpPolicy", env, verbose=1)
    #
    # # todo add evalutation env
    # model.learn(total_timesteps=100000)
    # model.save("subgoal_reach")

    # del model  # remove to demonstrate saving and loading

    model = PPO.load("subgoal_reach", env=env)
    # env = SubGoalEnv()
    obs = env.reset()
    total_mean = 0
    for i in range(20):
        print("start:", i)
        sum_distance = 0
        for j in range(20):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # env.render()
            # print(j, ' ', info)
            if info['success']:
                print(" finished with steps:", j)
                break
            sum_distance += info['near_object']
        print("mean distance:", sum_distance/20)
        total_mean += sum_distance/20
        env.reset()
    total_mean = total_mean/20
    print("\ntotal mean:", total_mean)


if __name__ == '__main__':
    subgoal_env_policy()