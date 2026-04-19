import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation



# --- Training Section ---
print("Training Section!")


def make_env():
    env = gym.make("LunarLander-v3", continuous= False)
    return env

vec_env = make_vec_env(make_env, n_envs=8)

model = A2C("MlpPolicy", vec_env)

# model.learn(total_timesteps = 50000, progress_bar=True)

# model.save("A2C_LunarLander")



# --- Simulation Section ---
print("Simulating Section!")

model = A2C.load("A2C_LunarLander")


def make_env_sim():
    env = gym.make("LunarLander-v3", continuous= False, render_mode= "human")
    return env

env = make_vec_env(make_env_sim, n_envs=1)

obs = env.reset()

for i in range(2000):
    
    action,_ = model.predict(obs, deterministic=True)

    obs, rewards, dones, info = env.step(action)
    
    if i%200 == 0:
        print(f"Step: {i}")


env.close()
print("Finished!")