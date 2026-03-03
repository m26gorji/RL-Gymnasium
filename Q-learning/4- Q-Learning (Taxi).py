import gymnasium as gym
import minigrid
import numpy as np 


env = gym.make('Taxi-v3', 
               render_mode="none") # None Or human


observation, info = env.reset(seed=42)


Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
alpha = 0.1
gamma = 0.99
epsilon = 0.99


for _ in range(20000):
   observation, info = env.reset()
   random_start = env.observation_space.sample()
   env.unwrapped.s = random_start
   observation = random_start
   
   terminated = False
   truncated = False

   while not (terminated or truncated):
      current_state = observation
      
      # 0: Move left
      # 1: Move down
      # 2: Move right
      # 3: Move up
      if np.random.rand() > epsilon:
         action = np.argmax(Q[current_state])
      else:
         action = env.action_space.sample()

      
      observation, reward, terminated, truncated, info = env.step(action)
      
      next_state = observation
      
      
      best_next_action = np.argmax(Q[next_state])
      
      Q[current_state, action] = Q[current_state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[current_state, action])
      
      
   epsilon = max(0.01, epsilon * 0.999)
   
   if (_) % 500 == 0:
      print(f"Episode: {_}")

###################################################################

env = gym.make('Taxi-v3', 
               render_mode="human") # None Or human


input("Press Enter to start the simulation.")


observation, info = env.reset()
for _ in range(10):
   print(f"Episode start: {_+1}")
   terminated = False
   truncated = False
   
   while not (terminated or truncated):
      action = np.argmax(Q[observation])
      observation, reward, terminated, truncated, info = env.step(action)
      

      if terminated or truncated:
         observation, info = env.reset()
         random_start = env.observation_space.sample()

         env.unwrapped.s = random_start
         observation = random_start
         
         
env.close()