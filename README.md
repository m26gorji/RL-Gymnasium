# 🤖 Reinforcement Learning: Q-Learning Collection

This folder contains my implementations of the **Q-Learning** algorithm on several classic environments from the [Gymnasium](https://gymnasium.farama.org/) library. Each project demonstrates the agent's ability to learn an optimal policy through trial and error.

---

## 🎮 Project Overviews

### 1. FrozenLake (Deterministic) ❄️
In this version, the ice is not slippery. The agent's movements are 100% predictable, making it a perfect introduction to the Q-Table update mechanism.

`1- Q-Learning (FrozenLake_deterministic).py`

<img src="1-%20FrozenLake_deterministic.gif" alt="FrozenLake Deterministic" width="400">

---

### 2. FrozenLake (Stochastic/Slippery) ❄️
Here, the `is_slippery` parameter is enabled. The agent might slide in a different direction than intended. The Q-Learning agent learns to find the safest path to avoid falling into holes.

`2- Q-Learning (FrozenLake_stochastic).py`

<img src="2-%20FrozenLake_stochastic.gif" alt="FrozenLake Stochastic" width="400">

---

### 3. Cliff Walking 🧗‍♂️
A grid-world challenge where the agent must reach the goal while avoiding a cliff. A single wrong step results in a high penalty (-100).

`3- Q-Learning (CliffWalking).py`

<img src="3-%20CliffWalking.gif" alt="Cliff Walking" width="400">

---

### 4. Taxi-v3 🚕
The agent must pick up a passenger at one location and drop them off at another. This environment tests the agent's ability to handle multiple objectives (Pick-up -> Navigation -> Drop-off).

`4- Q-Learning (Taxi).py`

<img src="4-%20taxi_agent.gif" alt="Taxi Agent" width="400">