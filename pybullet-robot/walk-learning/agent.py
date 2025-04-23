import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl.agents.soft_actor_critic import SoftActorCritic
from chainerrl import distribution
import numpy as np
import gym
import pybullet_envs


class PolicyNetwork(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal(scale=1.0)
        super(PolicyNetwork, self).__init__()        
        with self.init_scope():
            self.l1 = L.Linear(28, 128, initialW = w)
            self.l2 = L.Linear(128, 128, initialW = w)
            self.l3 = L.Linear(128, 128, initialW = w)
            self.l4m = L.Linear(128, 8, initialW = w)
            self.l4v = L.Linear(128, 8, initialW = w)

    def __call__(self, s):
        h = F.relu(self.l1(s))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        m = F.tanh(self.l4m(h))
        log_scale = F.tanh(self.l4v(h))
        log_scale = F.clip(log_scale, -20., 2.)
        v = F.exp(log_scale * 2)
        return chainerrl.distribution.SquashedGaussianDistribution(m, v)


class QFunction(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal(scale=1.0)
        super(QFunction, self).__init__()        
        with self.init_scope():
            self.l1 = L.Linear(36, 128, initialW = w)
            self.l2 = L.Linear(128, 128, initialW = w)
            self.l3 = L.Linear(128, 128, initialW = w)
            self.l4 = L.Linear(128, 1, initialW = w)

    def __call__(self, s, a):
        h = F.concat((s, a), axis=1)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return  self.l4(h)


def main():
    env = gym.make('AntBulletEnv-v0')
    env.render(mode='human')

    
    num_episodes = 30000  

    q_func1 = QFunction() 
    q_func2 = QFunction() 
    policy = PolicyNetwork() 
    optimizer_p = chainer.optimizers.Adam(alpha=1e-3)
    optimizer_q1 = chainer.optimizers.Adam(alpha=1e-3)
    optimizer_q2 = chainer.optimizers.Adam(alpha=1e-3)
    optimizer_p.setup(policy)
    optimizer_q1.setup(q_func1)
    optimizer_q2.setup(q_func2)

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    def burnin_action_func():return np.random.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    agent = SoftActorCritic(policy, q_func1, q_func2, optimizer_p, optimizer_q1, optimizer_q2,
                 replay_buffer, gamma=0.99,
                 replay_start_size=128,
                 update_interval=8, 
                 soft_update_tau=0.005,
                 phi=phi, gpu=None, minibatch_size=128,
                 initial_temperature=1.0,
                 temperature_optimizer=None,
                 act_deterministically=False,
                 burnin_action_func=burnin_action_func,
                 entropy_target=-env.action_space.low.size)

    outdir = 'result/'
    #agent.load(outdir + "agent_sac_ant_trained")
    reward = 0

    for episode in range(1, num_episodes + 1):  
        done = False
        obs = env.reset()     
        while not done:
            action = agent.act_and_train(obs, reward)  
            obs, reward, done, info = env.step(action)   
        agent.stop_episode_and_train(obs, reward, done)   
        if episode % 10 == 0:  
            print('Episode {0:4d}: statistics: {1}'.format(episode, agent.get_statistics())) 
        if episode % 100 == 0:   
            agent.save(outdir + 'agent_sac_ant_trained')

if __name__ == '__main__':
    main()

