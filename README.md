# Purpose

this repogitory is run to study reinforcement learning.
thus, we apply the tech to control Self-discipline system.

# contents

1. basic: that is the code to check fundamental reinforcement theology.
2. documents: that is the note of reinforcement-learning.
3. pole-problem: that is the code to try the feinforcement learning.

Using Environment

- Open Gym
- Or Gym

## problems

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
| ------- | ------- |
| 1st | 289.0 |
| 2nd | 295.0 |
| 3rd | 278.0 |

### arranging boxes

when using ddqn, ai agent can arrange boxes toward restricted space.

![alt text](image/arranging-boxes.png)

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
