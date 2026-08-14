# faceTracingRL

Reinforcement learning for a Crazyflie 2.1 that follows a person's face.

A [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) agent is trained in a PyBullet
simulation to keep a detected face centred in frame and at a constant apparent size —
that is, to hold station in front of a moving subject — while a six-ray lidar keeps it
clear of obstacles. A separate OpenCV pipeline produces the same observation signals
from a live webcam, so the trained policy can be driven by a real camera.

---

## How it works

The agent never sees pixels. The vision layer reduces a camera frame to three numbers
describing where the face is and how large it appears; the policy consumes those plus
six range readings. This keeps the observation space tiny and lets the same policy run
against either the simulator or a real camera feed.

### Observation space — `Box(9,)`

| Index | Signal | Range | Meaning |
|---|---|---|---|
| 0 | `error_x` | −1.5 … 1.5 | Horizontal offset of the face from frame centre |
| 1 | `error_y` | −1.5 … 1.5 | Vertical offset of the face from frame centre |
| 2 | `face_area` | 0 … 1 | Apparent face size, a proxy for distance |
| 3–8 | `lidar` | 0 … 1 | Hit fraction along front, back, left, right, up, down |

A lidar value of `1.0` means nothing was struck within the 2 m ray.

### Action space — `Box(4,)`

Continuous, each in `[-1, 1]`: `roll`, `pitch`, `yaw`, `height`. Roll corrects
horizontal error, height corrects vertical error, and pitch adjusts the apparent face
size by moving toward or away from the subject.

### Reward

```
reward = 1.0                          # alive bonus
       − (error_x² + error_y²)        # stay centred
       − 10 · (face_area − 0.1)²      # hold target distance
       − 5.0   if min(lidar) < 0.15   # obstacle proximity
       − 20.0  on termination
```

An episode runs 500 steps and terminates early if the face leaves the frame
(`|error| > 1.2`) or the drone is about to strike something (`min(lidar) < 0.05`).

---

## Quick start

### Docker (recommended)

```bash
docker build -t facetracingrl .
docker run --rm facetracingrl
```

This trains for 20,000 timesteps and writes `final_drone_model.zip`. To keep the
checkpoints and TensorBoard logs on the host:

```bash
docker run --rm -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/sac_drone_tensorboard:/app/sac_drone_tensorboard" \
                facetracingrl
```

The image is CPU-only by design — `MlpPolicy` trains faster on CPU than on GPU for a
network this small, so the build installs the CPU PyTorch wheel.

### Local

Requires Python 3.9+. Building `face-recognition` compiles dlib, which needs CMake and
a C++ toolchain.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python train.py
```

Runs SAC across one environment per CPU core (capped at 8) using `SubprocVecEnv`.
Checkpoints land in `logs/` every 1,000 timesteps. Press <kbd>Ctrl</kbd>+<kbd>C</kbd>
at any point — the current model is saved before exit.

Follow training live:

```bash
tensorboard --logdir ./sac_drone_tensorboard/
```

### Watch the trained policy

```bash
python test.py
```

Opens the PyBullet GUI and flies the saved policy against a moving target. Requires
`final_drone_model.zip` and a display.

### Run the vision pipeline

```bash
python my_vision.py
```

Reads the webcam, matches faces against a reference photo, and prints the
`error_x / error_y / area` triple that feeds the policy. Place a clear, front-facing
photo of the target at `me.jpg`, or point it elsewhere:

```bash
REFERENCE_IMAGE=/path/to/face.jpg CAMERA_INDEX=1 python my_vision.py
```

Needs a webcam and a display, so it runs on the host rather than in the container.

---

## Project structure

```
CrazyflieFollowEnv.py   Gymnasium environment: physics, lidar, reward
train.py                SAC training with parallel environments
test.py                 Loads a trained policy and renders it
my_vision.py            Webcam face tracking; emits policy observations
requirements.txt        Pinned dependencies
Dockerfile              Multi-stage CPU-only training image
final_drone_model.zip   Pre-trained policy
logs/                   Training checkpoints
sac_drone_tensorboard/  TensorBoard event files
```

---

## Notes

- **The drone body is cosmetic.** The simulation is kinematic — `stepSimulation()` is
  never called — so the drone's pose is computed analytically from the face-tracking
  error. Its collision shape is masked out of the lidar so it cannot detect itself.
- **`quadrotor.urdf` is optional.** Recent `pybullet_data` releases no longer ship it;
  the environment falls back to a simple box, which affects appearance only.
- **`me.jpg` is not in the repository.** Supply your own reference photo before running
  `my_vision.py`.
- **The environment is fully seeded.** `reset(seed=...)` gives reproducible episodes,
  which matters when comparing training runs.
