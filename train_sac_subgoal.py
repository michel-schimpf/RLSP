from stable_baselines3 import SAC
from SubGoalEnv08042022 import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor


def train():
    algo = "SAC"
    ALGO = SAC
    models_dir = f"models/{algo}"
    logdir = "logs"
    TIMESTEPS = 512

    env = SubGoalEnv("pick-place-v2")
    env_vec = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             ])
    env_vec = VecMonitor(env_vec, "logs/SAC_0")
    model = ALGO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir,)
    # model = ALGO.load("models/SAC/1220000", env=env_vec)
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                    tb_log_name=algo, log_interval=512)
        model.save(f"{models_dir}/{TIMESTEPS * iters}")


if __name__ == '__main__':
    train()


