import sys
sys.path.append('/home/lwx/Reinforcement_Learning')

import os 
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from DroneProject.envs.DroneWaypointAviary import DroneWaypointAviary
from DroneProject.utils.utils import sync

DEFAULT_GUI = True

def run(gui=DEFAULT_GUI):

    test_env = DroneWaypointAviary(gui=gui)
    
    zip_location = "/home/lwx/Reinforcement_Learning/DroneProject/RL/results/save-07.11.2024_15.47.45"
    model_file = "best_model.zip"
    model = PPO.load(zip_location + "/" + model_file)

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    for i in range((test_env.EPISODE_LEN_SEC+10)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})

    

    test_env.close()
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', test_env.action_space)
    print('[INFO] Observation space:', test_env.observation_space)

if __name__== '__main__':
    run(DEFAULT_GUI)