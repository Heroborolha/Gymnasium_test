import numpy as np
import gymnasium as gym
from typing import SupportsFloat
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Discrete

class RelativePosition(ObservationWrapper):

    def __init__(self, env):
        super().__init__()
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        return obs["target"] - obs["agent"]
    
class DiscreteActions(ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]   
    
if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    wrapped_env = DiscreteActions(env, [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])
    print(wrapped_env.action_space)

class ClipRewards(RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return np.clip(r, self.min_reward, self.max_reward)   
    
class ReacherRewardWrapper(Wrapper):
    def __init__(self, env, reward_dist_weight, reward_ctrl_weight):
        super().__init__(env) 
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = (self.reward_dist_weight * info["reward_dist"] + self.reward_ctrl_weight * info["reward_ctrl"])
        return obs, reward, terminated, truncated, info