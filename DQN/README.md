# Deep Q-Network (DQN) Implementation 🧠

This folder contains Deep Q-Network (DQN) implementations using **PyTorch** and **Gymnasium**. These agents are trained to solve classic control and arcade-style environments by approximating the optimal action-value function.

---

## 🎮 Environments & Results

### 1. CartPole-v1
The goal is to prevent the pole from falling by moving the cart left or right. The agent receives a reward of +1 for every step the pole remains upright.

<img src="images/CartPole.gif" alt="CartPole" width="400">


---

### 2. Flappy Bird
The agent learns to navigate a bird through gaps between pipes. This implementation uses the `flappy-bird-gymnasium` environment.


<img src="images/FlappyBird.gif" alt="FlappyBird" width="200">


---

## 📂 File Descriptions

| File | Description |
| :--- | :--- |
| `dqn.py` | Basic DQN architecture with a single hidden layer. |
| `dqn2layers.py` | Enhanced DQN architecture with two hidden layers for more complex tasks. |
| `CartPole with DQN.py` | Training and testing script specifically for the CartPole environment. |
| `FlappyBird with DQN.py` | Training and testing script specifically for the Flappy Bird environment. |
| `Cartpole_model.pth` | Pre-trained weights for the CartPole agent. |
| `FlappyBird_model.pth` | Pre-trained weights for the Flappy Bird agent. |

---

## 🛠 Features
* **Experience Replay:** Uses a buffer to break correlations between consecutive samples, significantly improving convergence.
* **Target Network:** Implements a separate target network (updated periodically) to provide stable Q-value targets during training.
* **Epsilon-Greedy Policy:** Balances exploration (random actions) and exploitation (best-known actions).
* **Custom Reward Shaping:** Optimized reward signals to help the agent learn faster in competitive environments.