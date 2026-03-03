import torch
from torch import nn
import gymnasium as gym
import time
import flappy_bird_gymnasium
from dqn2layers import DQN
from experience_reply import ReplyMemory
import random
import os


start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"


reply_emory_size = 10000
mini_batch_size = 64
epsilon_init = 1
epsilon_decay = 0.9997
epsilon_min = 0.05
network_syn_rate = 1000
learning_rate = 0.0001
discount_factor = 0.99
optimizer = None
hidden_layers = 256


class Agent:
    
    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions, hidden_layers).to(device)
        

        epsilon = 0.0
        
        if is_training:
            memory = ReplyMemory(reply_emory_size)
            epsilon = epsilon_init
            target_dqn = DQN(num_states, num_actions, hidden_layers).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_counter = 0
            optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)



        previous_episodes = 0
        
        if os.path.exists("FlappyBird_model.pth"):
            checkpoint = torch.load("FlappyBird_model.pth")
            
            policy_dqn.load_state_dict(checkpoint['model_state_dict'])
            
            previous_episodes = checkpoint.get('episode_count', 0)
            current_epsilon = checkpoint.get('epsilon', epsilon_init)
            
            print(f"Number of previous episodes completed = {previous_episodes}")
            print(f"The current epsilon is = {current_epsilon}")
            
            if is_training:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                epsilon = current_epsilon
                print("Previous weights have been loaded.")
                print('-------------------------------')
                
                
                
        episode = 5
        
        if is_training:
            episode = 3000
        
        
        rewards_per_episode =[]
        epsilon_history =[]
        
        for e in range(episode):
                        
            if not is_training:
                print(f"Episode: {e}")

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            truncated = False
            episode_reward = 0.0
            
            while True:
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device)
                    
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                
                if not terminated:
                  actual_reward = 0.1 

                  if reward > 0:
                      actual_reward = 10.0 
                else:
                  actual_reward = -5.0
                  
                reward = torch.tensor(actual_reward, dtype=torch.float, device=device)
                    
                
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_counter += 1
                    
                    
                    if len(memory) > mini_batch_size:
                        mini_batch = memory.sample(mini_batch_size)
                        
                        self.optimize(mini_batch, policy_dqn, target_dqn, optimizer)
                        
                        if step_counter > network_syn_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_counter = 0
                            
                            # Saving data
                            checkpoint = {
                                'model_state_dict': policy_dqn.state_dict(),
                                'episode_count': previous_episodes + e + 1,
                                'epsilon': epsilon 
                            }
                            torch.save(checkpoint, "FlappyBird_model.pth")
                            
                
                
                state = new_state
                
                episode_reward += reward.item()
                
                if render:
                    env.render()
                
                if not is_training:
                    time.sleep(0.02)

    
    
                if terminated:
                    # print(memory.memory[-1])
                    # print(len(memory))
                    # print('---------------')
                    break
                
            epsilon = max(epsilon*epsilon_decay, epsilon_min)
            rewards_per_episode.append(episode_reward)
            epsilon_history.append(epsilon)
            
            if is_training:
                if (e) % 500 == 0:
                    current_time = time.time()
                    total_duration = current_time - start_time
                    avg_episode_time = total_duration / (e + 1)
                    remaining_episodes = episode - (e + 1)
                    remain_time = avg_episode_time * remaining_episodes
                    rem_min, rem_sec = divmod(remain_time, 60)
                    print(f"Time Remaining: {int(rem_min)}m {int(rem_sec)}s")
                    print(f"Episode: {e} from {episode}")
                    print('-------------------------------')
            

                
        env.close()
        
    
    def optimize(self, mini_batch, policy_dqn, target_dqn, optimizer):

        states, actions, new_states, rewards, terminations = zip(*mini_batch)
            
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
            
            
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * discount_factor * target_dqn(new_states).max(dim=1)[0]
                
        
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        loss = torch.nn.functional.mse_loss(current_q, target_q)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("Starting Training...")
agent = Agent()
# agent.run(is_training=True, render=False)


print("Starting Test (Visual)...")
agent.run(is_training=False, render=True)