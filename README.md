# [Purpose](https://github.com/Shinichi0713/Reinforce-Learning-Study)

this repogitory is run to study reinforcement learning.
thus, we apply the tech to control Self-discipline system.

![1762658300029](image/README/main_title.png)

# contents

1. basic: that is the code to check fundamental reinforcement theology.
2. documents: that is the note of reinforcement-learning.
3. pole-problem: that is the code to try the feinforcement learning.

Using Environment

- Open Gym
- Or Gym
- Muti Agent



# my work

I am writing a book on reinforcement learning.

The book is designed for beginners to learn reinforcement learning step by step, covering everything from the basics to practical applications.

It is published on Amazon Kindle, so please feel free to check it out if you are interested.

[Amazon.co.jp: 実践!強化学習入門: Pythonで動かしながら理解する AI学習書 (AI関係書籍) eBook : 3 Sons Lover: Kindleストア](https://www.amazon.co.jp/%E5%AE%9F%E8%B7%B5-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E5%85%A5%E9%96%80-Python%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%97%E3%81%AA%E3%81%8C%E3%82%89%E7%90%86%E8%A7%A3%E3%81%99%E3%82%8B-AI%E5%AD%A6%E7%BF%92%E6%9B%B8-AI%E9%96%A2%E4%BF%82%E6%9B%B8%E7%B1%8D-ebook/dp/B0FH4VKHLJ)

<img src="image/README/1763209669056.png" alt="q-learn" width="300px" height="auto">


## My Codes

The repository of my scratch code is stored in next URL:

[Shinichi0713/Reinforce-Learning-Study: this is the codes which is in accordance with reinforcement-learning](https://github.com/Shinichi0713/Reinforce-Learning-Study)

On the other hand, I have LLM repository also.

[Shinichi0713/LLM-fundamental-study: this site is the fundamental page of LLM-mechanism](https://github.com/Shinichi0713/LLM-fundamental-study)

Please look.

## problems

### Gird World with Dyna-Q

with using Dyna-Q, train agent to update the model.

below shows trainstion of reward vs episode.

![1762658300029](image/README/1762658300029.png)

### ball cather

this is the behavior of q-learning agent.

<img src="image/ball-catch-q-agent.gif" alt="q-learn" width="300px" height="auto">

### pole cart

with using dqn, the motion is completed.

<img src="image/pole-cart.gif" alt="q-learn" width="300px" height="auto">

### pendulum

this is the behavior of SAC.

<img src="image/pendulum.gif" alt="q-learn" width="300px" height="auto">

### luna-landing

this is the behavior of SAC.
DDPG can't work well.

<img src="image/luna-landing.gif" alt="sac" width="300px" height="auto">

### robo-walking

this is the behavior of actor-critic.

<img src="image/robo-walking.gif" alt="sac" width="300px" height="auto">

### BipedalWalkerHardcore

with using sac, the agent gradually walk...

the agent of this walker is based on just fnn model.
essencially, the progress of train isn't proceed well.

<img src="image/bipedal_walker_v1.gif" alt="sac" width="300px" height="auto">

at the next, the agent is composed based on transformer.
this agent size isn't large.
but, the progress of train proceed as expected.
so that, i find ,in RL , the architecture is important.
unfortunately, the agent don't use both legs.
this would be owing to short of exploration.

<img src="image/bipedalwalkerhardcore.gif" alt="sac" width="300px" height="auto">

<img src="image/bipewalker_another.gif" alt="sac" width="300px" height="auto">

### TSP

this is the behavior of PointerNet.
not good....

<img src="image/TSP.png" alt="sac" width="300px" height="auto">

this is the result with using 3 methods.

<img src="image/TSP-2.png" alt="sac" width="300px" height="auto">

### JSSP

using Actor-critc framework.
and, the model is composed with Transformer Network.
learn how short the total job become.

<img src="image/JSSP-1.png" alt="jssp-1" width="300px" height="auto">

<img src="image/JSSP-2.png" alt="jssp-2" width="300px" height="auto">

<img src="image/JSSP-3.png" alt="jssp-3" width="300px" height="auto">

### imitation learning - behavior clone

when imitation learning is utilized, i check the effect.
in this case, reward is improved when using imitation learning.

<img src="image/reward_history_imitation_0.png" alt="jssp-1" width="300px" height="auto">

<img src="image/reward_history_imitation_50.png" alt="jssp-2" width="300px" height="auto">

<img src="image/reward_history_imitation_300.png" alt="jssp-3" width="300px" height="auto">

### IRL-GAIL

with using stable_baselines3 and imitation, agent is trained with gail.
the reward is given as below.

| trial no | reward |
| -------- | ------ |
| 1st      | 289.0  |
| 2nd      | 295.0  |
| 3rd      | 278.0  |

### arranging boxes

when using ddqn, ai agent can arrange boxes toward restricted space.

![alt text](image/arranging-boxes.png)

### Multi-Agent example

We implemented DQN training using two Rock-Paper-Scissors agents as a multi-agent example problem. We visualized two aspects:

The trend of Agent 1's average reward (Learning stability).

The trend of Agent 1's final Q-values (Learned action strategy).

<img src="image/README/1763209192634.png" alt="jssp-3" width="500px" height="auto">

#### MARL warehouse

new theme is considering.
the environment is displayed as next.

<img src="image/README/marl_agent_motion.gif" alt="jssp-3" width="500px" height="auto">

#### MARL adventure

This is a cooperative Multi-Agent Reinforcement Learning (MARL) example focusing on **information sharing** and **continuous coordination**. The core challenge is to efficiently cover an unknown area by pooling decentralized knowledge.

__Environment and Setup__

| Item | Details |
| :--- | :--- |
| **Environment** | An **unknown grid map** representing a disaster site where critical targets are hidden. |
| **Observation** | Each drone has a **very narrow sensor range** (e.g., only adjacent cells), leading to significant local **partial observability**. |
| **Agents** | Multiple search drones (or mobile sensor robots). |
| **Actions** | Movement (Up, Down, Left, Right, Stay). |
| **Goal** | **Maximize map coverage efficiency** by minimizing the time required to fully explore the entire map (minimizing unexplored area). |

__Learning Objectives and Cooperation Points__

1. Information Sharing and Distributed Knowledge

* **Necessity for Coordination:** Without sharing information about previously explored areas, agents will inefficiently perform **redundant searches**.
* **Learning Goal:** Agents must learn to integrate their local observations into a **common global knowledge map (shared memory)** and use this map to choose a strategy that prioritizes moving toward **unexplored locations**.

2. Optimal Coverage and Spatial Load Balancing

* **Nature of Coordination:** This task emphasizes **positive cooperation** ("dividing up the unexplored area") rather than negative cooperation ("avoiding collisions").
* **Learning Goal:** The drone swarm must learn a **spatial load-balancing strategy**: maintaining an **appropriate distance** from each other to avoid overcrowding while cooperatively segmenting the search area to maximize the overall coverage rate.

<img src="image/README/marl_agent_adventure.gif" alt="jssp-3" width="500px" height="auto">

# cite

in this repogitory, oss 'pygame-learning-environment' is used.
[https://github.com/ntasfi/PyGame-Learning-Environment](https://github.com/ntasfi/PyGame-Learning-Environment)

deep mind archives!

very nice site!

[google-deepmind/deepmind-research: This repository contains implementations and illustrative code to accompany DeepMind publications](https://github.com/google-deepmind/deepmind-research/tree/master)

# References

when studying RL, I refer to any other web-site.
show the reference site.

[AI compass](https://ai-compass.weeybrid.co.jp/)
this site indicates many ai knowledge with insight.

[星の本棚](https://yagami12.hatenablog.com/entry/2019/02/22/210608)
this site shows nice tips about reinforcement learning.
