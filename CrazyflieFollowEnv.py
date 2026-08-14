import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

# Observation layout: [error_x, error_y, face_area, lidar_f, lidar_b, lidar_l, lidar_r, lidar_u, lidar_d]
# error_x/error_y can overshoot +-1.0 by one step's worth of correction before the
# episode terminates at +-1.2, so the box is widened to keep observations legal.
OBS_LOW = np.array([-1.5, -1.5, 0.0] + [0.0] * 6, dtype=np.float32)
OBS_HIGH = np.array([1.5, 1.5, 1.0] + [1.0] * 6, dtype=np.float32)

# The drone body is cosmetic (physics is never stepped) but it still owns a
# collision shape, so the lidar rays -- which start at the drone -- were hitting
# it and reporting phantom obstacles. Clearing the body's own collision mask
# takes it out of every ray test. Note this must be done body-side: masking the
# rays instead would also drop the plane and the obstacles, which Bullet puts in
# the StaticFilter group rather than the default one.
DRONE_GROUP = 0b10


class CrazyflieFollowEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(CrazyflieFollowEnv, self).__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}, expected one of "
                f"{self.metadata['render_modes']} or None"
            )
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

        self.state = None
        self.target_area = 0.1
        self.steps_left = 0

        self.person_world_pos = np.array([0.0, 0.0, 1.0])
        self.person_velocity = np.array([0.0, 0.0])
        self.obstacle_ids = []

        self.physics_client = None
        self.drone_id = None
        self.face_id = None

        connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT

        # Every pybullet call below is scoped to this client id. Without it the
        # calls silently target client 0, so two envs in one process (DummyVecEnv)
        # would stomp on each other's world.
        self.physics_client = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        face_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0.8, 0.6, 1],
            physicsClientId=self.physics_client,
        )
        self.face_id = p.createMultiBody(
            baseVisualShapeIndex=face_visual,
            basePosition=self.person_world_pos,
            physicsClientId=self.physics_client,
        )

        try:
            self.drone_id = p.loadURDF(
                "quadrotor.urdf", [0, -0.5, 1], globalScaling=0.5,
                physicsClientId=self.physics_client,
            )
        except p.error:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], rgbaColor=[0, 0, 1, 1],
                physicsClientId=self.physics_client,
            )
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02],
                physicsClientId=self.physics_client,
            )
            self.drone_id = p.createMultiBody(
                baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                basePosition=[0, -0.5, 1],
                physicsClientId=self.physics_client,
            )

        self._exclude_from_lidar(self.drone_id)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=90, cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.physics_client,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.person_world_pos = np.array([0.0, 0.0, 1.0])
        # self.np_random is seeded by super().reset(); using the global np.random
        # here would make the env ignore its seed entirely.
        self.person_velocity = self.np_random.uniform(-0.01, 0.01, size=2)

        for obs in self.obstacle_ids:
            p.removeBody(obs, physicsClientId=self.physics_client)
        self.obstacle_ids = []

        for _ in range(5):
            x_pos = self.np_random.choice([-1, 1]) * self.np_random.uniform(1.0, 2.5)
            y_pos = self.np_random.choice([-1, 1]) * self.np_random.uniform(1.0, 2.5)

            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5], rgbaColor=[0.4, 0.4, 0.4, 1],
                physicsClientId=self.physics_client,
            )
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5],
                physicsClientId=self.physics_client,
            )
            obs_id = p.createMultiBody(
                baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                basePosition=[x_pos, y_pos, 0.5],
                physicsClientId=self.physics_client,
            )
            self.obstacle_ids.append(obs_id)

        random_x = self.np_random.uniform(-0.5, 0.5)
        random_y = self.np_random.uniform(-0.5, 0.5)
        random_area = self.np_random.uniform(0.05, 0.3)

        self.state = np.clip(
            np.array(
                [random_x, random_y, random_area, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            OBS_LOW,
            OBS_HIGH,
        )
        self.steps_left = 500

        self._sync_bodies(self.person_world_pos, [0.0, 0.0, 0.0, 1.0])

        return self.state, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        roll, pitch, _yaw, height_cmd = action

        current_x, current_y, current_area = self.state[:3]

        accel = self.np_random.uniform(-0.002, 0.002, size=2)
        self.person_velocity = (self.person_velocity + accel) * 0.98
        self.person_velocity = np.clip(self.person_velocity, -0.03, 0.03)

        person_dx = self.person_velocity[0]
        person_dy = self.person_velocity[1]

        self.person_world_pos[0] += person_dx
        self.person_world_pos[1] += person_dy
        self.person_world_pos[0] = np.clip(self.person_world_pos[0], -3, 3)
        self.person_world_pos[1] = np.clip(self.person_world_pos[1], -3, 3)

        correction_x = (roll * 0.05)
        correction_y = (height_cmd * 0.05)
        correction_area = (pitch * 0.005)

        new_x = current_x - correction_x + person_dx
        new_y = current_y - correction_y
        new_area = current_area + correction_area - (person_dy * 0.1)
        new_area = max(0.05, min(0.8, new_area))

        dist_from_face = 1.0 / (new_area + 0.1) * 0.1
        drone_world_x = self.person_world_pos[0] - new_x
        drone_world_y = self.person_world_pos[1] - dist_from_face
        drone_world_z = self.person_world_pos[2] + new_y

        lidar_readings = self._read_lidar([drone_world_x, drone_world_y, drone_world_z])

        distance_penalty = -(new_x ** 2 + new_y ** 2)
        size_penalty = -((new_area - self.target_area) ** 2) * 10

        obstacle_penalty = 0
        min_dist = min(lidar_readings)
        if min_dist < 0.15:
            obstacle_penalty = -5.0

        reward = distance_penalty + size_penalty + obstacle_penalty + 1.0

        self.state = np.clip(
            np.array([new_x, new_y, new_area] + lidar_readings, dtype=np.float32),
            OBS_LOW,
            OBS_HIGH,
        )

        if self.render_mode == "human":
            tilt = p.getQuaternionFromEuler(
                [pitch * 0.4, roll * 0.4, 0], physicsClientId=self.physics_client
            )
            self._sync_bodies(
                self.person_world_pos, tilt,
                drone_pos=[drone_world_x, drone_world_y, drone_world_z],
            )
            time.sleep(1 / self.metadata["render_fps"])

        self.steps_left -= 1
        terminated = False
        truncated = self.steps_left <= 0

        # Bounds are checked against the raw values, before the observation clip.
        if abs(new_x) > 1.2 or abs(new_y) > 1.2 or min_dist < 0.05:
            terminated = True
            reward -= 20

        return self.state, reward, terminated, truncated, {}

    def _read_lidar(self, start_pos, ray_len=2.0):
        directions = [
            [0, ray_len, 0],   # Front
            [0, -ray_len, 0],  # Back
            [-ray_len, 0, 0],  # Left
            [ray_len, 0, 0],   # Right
            [0, 0, ray_len],   # Up
            [0, 0, -ray_len],  # Down
        ]

        end_positions = [
            [start_pos[0] + d[0], start_pos[1] + d[1], start_pos[2] + d[2]]
            for d in directions
        ]
        # One batched call instead of six round trips to the physics server.
        results = p.rayTestBatch(
            [start_pos] * len(end_positions), end_positions,
            physicsClientId=self.physics_client,
        )
        return [float(r[2]) for r in results]

    def _exclude_from_lidar(self, body_id):
        """Move a body out of the world collision group so rays pass through it."""
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        for link in range(-1, num_joints):
            p.setCollisionFilterGroupMask(
                body_id, link,
                collisionFilterGroup=DRONE_GROUP,
                collisionFilterMask=0,
                physicsClientId=self.physics_client,
            )

    def _sync_bodies(self, face_pos, drone_orn, drone_pos=None):
        """Move the visual bodies to match the internal state (no-op in DIRECT mode)."""
        if self.render_mode != "human" or self.physics_client is None:
            return
        if self.face_id is not None:
            p.resetBasePositionAndOrientation(
                self.face_id, face_pos, [0, 0, 0, 1],
                physicsClientId=self.physics_client,
            )
        if self.drone_id is not None and drone_pos is not None:
            p.resetBasePositionAndOrientation(
                self.drone_id, drone_pos, drone_orn,
                physicsClientId=self.physics_client,
            )

    def render(self):
        # GUI mode renders continuously from step(); nothing to do here.
        return None

    def close(self):
        # `if self.physics_client:` was a bug: the first client id is 0, which is
        # falsy, so the default env never disconnected.
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except p.error:
                pass  # Already disconnected.
            self.physics_client = None
