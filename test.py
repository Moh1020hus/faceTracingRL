import os
import sys

from stable_baselines3 import SAC

from CrazyflieFollowEnv import CrazyflieFollowEnv

MODEL_PATH = "final_drone_model.zip"


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found: {MODEL_PATH}. Run train.py first.")

    env = CrazyflieFollowEnv(render_mode="human")
    try:
        model = SAC.load(MODEL_PATH, device="cpu")

        obs, _ = env.reset()
        print("Test started! Press Ctrl+C to stop.")

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Without this the pybullet GUI client leaks on every exit path.
        env.close()


if __name__ == "__main__":
    main()
