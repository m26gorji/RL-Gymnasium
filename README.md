# Reinforcement Learning Collection: DQN & Q-Learning

This repository contains implementations of various Reinforcement Learning (RL) algorithms applied to classic control and arcade environments using **OpenAI Gymnasium**. The project is divided into Deep Q-Networks (DQN) for continuous/complex state spaces and Tabular Q-Learning for discrete environments.

---

## 📂 Project Structure

### 1. Deep Q-Network (DQN)
Located in the `DQN/` directory, this section utilizes Deep Neural Networks.

* **Environments:** 
    * `CartPole-v1`: Balancing a pole on a cart.
    * `FlappyBird-v0`: Navigating a bird through pipes (via `flappy-bird-gymnasium`).
* **Core Files:**
    * `dqn.py`: Standard Deep Q-Network architecture.
    * `dqn2layers.py`: Enhanced architecture with deeper layers.
    * `*.pth`: Pre-trained PyTorch models for immediate testing.



---

### 2. Q-Learning (Tabular)
Located in the `Q-learning/` directory, these scripts implement the classic Q-Learning algorithm using a Q-Table to solve discrete state-space problems.

* **Environments:**
    * `FrozenLake-v1`: Navigating a grid (Deterministic and Stochastic versions).
    * `CliffWalking-v0`: Finding the safest path along a cliff.
    * `Taxi-v3`: Efficiently picking up and dropping off passengers.