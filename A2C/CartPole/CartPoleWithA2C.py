from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import time


# --- Training Section ---
print("Training Section!")

vec_env = make_vec_env("CartPole-v1", n_envs=16)

model = A2C("MlpPolicy", vec_env)

# model.learn(total_timesteps = 100000)

# model.save("A2C_CartPole")



# --- Simulation Section ---
print("Simulating Section!")

model = A2C.load("A2C_CartPole")


env = make_vec_env("CartPole-v1", n_envs=1, env_kwargs={"render_mode": "human"})
obs = env.reset()

for i in range(10):
    
    action,_ = model.predict(obs, deterministic=True)
    
    obs, rewards, dones, info = env.step(action)
    time.sleep(0.02)
    # env.render("human")
    
    if i%100 == 0:
        print(f"Step: {i}")


env.close()
print("Finished!")



# import imageio

# env = make_vec_env("CartPole-v1", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
# obs = env.reset()


# frames = []
# for i in range(500):

#     action,_ = model.predict(obs, deterministic=True)
    
#     obs, rewards, dones, info = env.step(action)
#     time.sleep(0.02)
    
#     frames.append(env.render())


#     if i%100 == 0:
#         print(f"Step: {i}")

# imageio.mimsave('A2C_CartPole.gif', frames, fps=50)
# print("GIF file saved successfully!")

# env.close()