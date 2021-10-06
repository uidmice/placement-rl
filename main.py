import gym
from gym.envs.registration import register


register(id='Placement-v0', entry_point='env.placement_env:PlacementEnv')
env = gym.make('Placement-v0')