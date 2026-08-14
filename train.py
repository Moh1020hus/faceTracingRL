import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from CrazyflieFollowEnv import CrazyflieFollowEnv

TOTAL_TIMESTEPS = 20000
CHECKPOINT_EVERY = 1000  # In total timesteps, across all envs.
MAX_ENVS = 8  # Beyond this the per-env pybullet servers cost more than they gain.


def main():
    # os.cpu_count() returns None on some platforms, and spawning one pybullet
    # server per core gets counterproductive on big machines.
    num_cpu = min(os.cpu_count() or 1, MAX_ENVS)
    print(f"Creating {num_cpu} parallel environments...")

    env = make_vec_env(
        CrazyflieFollowEnv,
        n_envs=num_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": None},
    )

    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.001,
            tensorboard_log="./sac_drone_tensorboard/",
            device="cpu",  # MlpPolicy is faster on CPU than GPU.
        )

        # CheckpointCallback counts calls, not total timesteps, so save_freq has
        # to be divided by n_envs or it fires num_cpu times too often.
        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_EVERY // num_cpu, 1),
            save_path="./logs/",
            name_prefix="sac_drone",
        )

        print("-----------------------------------------")
        print("STARTING PARALLEL TRAINING... Press Ctrl+C to stop")
        print("-----------------------------------------")

        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
        except KeyboardInterrupt:
            # The banner tells the user to press Ctrl+C; without this the whole
            # run was discarded instead of saved.
            print("\nInterrupted - saving current model...")

        model.save("final_drone_model")
        print("Model saved to final_drone_model.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
