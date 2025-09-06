import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SectionEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.track1 = 0
        self.track2 = 0
        self.trainA_priority = np.random.choice([0.0, 1.0])
        self.trainB_priority = np.random.choice([0.0, 1.0])
        self.timeA = np.random.uniform(0.2, 1.0)
        self.timeB = np.random.uniform(0.2, 1.0)
        self.timestep = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.track1, self.track2,
                         self.trainA_priority, self.trainB_priority,
                         self.timeA, self.timeB], dtype=np.float32)

    def step(self, action):
        self.timestep += 1
        reward = -0.05
        done = False

        if action == 1:
            if self.track1 == 0 and self.timeA < 0.5:
                self.track1 = 1
                reward += 1.0
            else:
                reward -= 0.5
        elif action == 2:
            if self.track2 == 0 and self.timeB < 0.5:
                self.track2 = 1
                reward += 1.0
            else:
                reward -= 0.5
        else:
            reward -= 0.02

        self.timeA = max(0.0, self.timeA - 0.1)
        self.timeB = max(0.0, self.timeB - 0.1)

        if self.track1 == 1: self.track1 = 0
        if self.track2 == 1: self.track2 = 0

        if self.timestep >= 100:
            done = True

        return self._get_obs(), reward, done, False, {}
