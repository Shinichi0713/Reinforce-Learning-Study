# Puepose

This site is summary of Multi-Agent RL, shows the results of MARL trials.



## *Shared Brain Deep Q-Network (Shared Brain DQN)

This is an implementation of a **Shared Brain Deep Q-Network (Shared Brain DQN)** aimed at solving the cooperative challenge of **multi-sensor (agent) search and exploration in a disaster area**.

The learning content of this code is centered on establishing a reinforcement learning process to achieve **efficient exploration and coordination**.

## Summary of Learning Content

The agents primarily learn and optimize three key areas in this code:

### 1. Shared Knowledge-Based Decision Making

* **Learning Goal**: Agents learn a strategy that considers the **knowledge of areas already explored by other agents** to decide "where to move next."
* **Implementation**:
  * **State Input**: The input to the agent (`preprocess_state`) includes both the **Shared Exploration Map (Ch0)** and the agent's own position (Ch1). This allows the network to select actions based on **global exploration progress** in addition to local information.
  * **SharedAgent**: All agents **share the weights of a single policy network** ($\text{policy\_net}$). Consequently, the learning outcome of one agent is instantaneously reflected across the entire team, achieving **implicit knowledge sharing**.

### 2. Spatial Load Balancing (Distributed Exploration)

* **Learning Goal**: Agents learn a spatial coordination strategy to move in a way that **avoids clustering together** and instead **divides and covers unexplored, new areas.**
* **Implementation**:
  * **Reward Design**: The reward is given based on the total area of **newly explored cells** by the team (assuming this is how `env.step` is designed). This discourages redundant exploration and guides **distributed actions** toward higher rewards.
  * **CNN Utilization**: The **$\text{DuelingDQN}$** uses a **$\text{CNN}$** to recognize **spatial patterns** like "large unexplored clusters" or "agent congregation spots" from the input map image, learning to estimate higher Q-values for actions leading toward unexplored directions.

### 3. Stable Learning via DQN Mechanisms

* **Learning Goal**: To stabilize Q-value estimation and ensure efficient convergence in a complex exploration task.
* **Implementation**:
  * **Experience Replay ($\text{ReplayMemory}$)**: Past experiences are sampled randomly, which breaks the correlation in the data, enhancing learning stability.
  * **Target Network ($\text{target\_net}$)**: Using the weights of an older network for the Q-value target calculation prevents **divergence due to self-reference** during training.
  * **$\epsilon$-greedy Strategy**: Balances the selection between random actions (exploration) in the early stages of learning and utilizing learned knowledge (optimal action) as training progresses.

## Conclusion

This code learns a strategy where multiple drones complete the disaster site exploration **efficiently and non-redundantly** by optimizing the weights of a **single DQN network** under a **reward system that prioritizes cooperative actions**, using **"shared observations and position information"** as input.

![1764464138183](image/readme/1764464138183.png)

<img src="image/README/marl_agent_adventure.gif" alt="jssp-3" width="500px" height="auto">

<img src="image/README/marl_adventure.gif" alt="jssp-3" width="500px" height="auto">
