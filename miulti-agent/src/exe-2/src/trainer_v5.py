import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import collections

# --- ハイパーパラメータ ---
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # エントロピー正則化係数
BUFFER_LIMIT = 50000
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 観測の変換関数 ---
def flatten_obs(obs_dict, env_size=GRID_SIZE):
    """(pos, holding, orders) -> flat vector"""
    flats = {}
    for i, (pos, holding, orders) in obs_dict.items():
        # 正規化された位置
        pos_norm = [pos[0] / env_size, pos[1] / env_size]
        # 荷物持ちフラグ
        h_flag = [1.0 if holding else 0.0]
        # 残りオーダー（固定長フラグ化）
        order_flags = [0.0] * NUM_ORDERS
        for o in orders:
            order_flags[o] = 1.0
        
        flats[i] = torch.FloatTensor(pos_norm + h_flag + order_flags).to(device)
    return flats

# --- モデル定義 ---
class Actor(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions) # 各アクションのQ値を出力
        )

    def forward(self, x):
        return self.fc(x)

# --- SAC エージェント ---
class SACAgent:
    def __init__(self, n_obs, n_actions):
        self.actor = Actor(n_obs, n_actions).to(device)
        self.critic_1 = Critic(n_obs, n_actions).to(device)
        self.critic_2 = Critic(n_obs, n_actions).to(device)
        self.target_critic_1 = Critic(n_obs, n_actions).to(device)
        self.target_critic_2 = Critic(n_obs, n_actions).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=LR_CRITIC)
        self.n_actions = n_actions

    def get_action(self, obs):
        probs = self.actor(obs)
        dist = Categorical(probs)
        return dist.sample().item()

    def train(self, transitions):
        s, a, r, s_prime, done = zip(*transitions)
        s = torch.stack(s); a = torch.tensor(a).unsqueeze(1).to(device)
        r = torch.tensor(r).unsqueeze(1).to(device); s_prime = torch.stack(s_prime)
        done = torch.tensor(done).unsqueeze(1).to(device)

        # Critic Update
        with torch.no_grad():
            next_probs = self.actor(s_prime)
            next_log_probs = torch.log(next_probs + 1e-8)
            target_q1 = self.target_critic_1(s_prime)
            target_q2 = self.target_critic_2(s_prime)
            target_v = torch.sum(next_probs * (torch.min(target_q1, target_q2) - ALPHA * next_log_probs), dim=1, keepdim=True)
            target_q = r + GAMMA * (1 - done.float()) * target_v

        q1 = self.critic_1(s).gather(1, a)
        q2 = self.critic_2(s).gather(1, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        q1 = self.critic_1(s)
        q2 = self.critic_2(s)
        min_q = torch.min(q1, q2)
        
        actor_loss = torch.sum(probs * (ALPHA * log_probs - min_q), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Target Update
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

# --- メイン学習ループ ---
env = WarehouseEnv()
n_obs_flat = 2 + 1 + NUM_ORDERS # pos(2) + holding(1) + orders(3) = 6
agents = {i: SACAgent(6, env.action_space) for i in range(NUM_AGENTS)}
memory = {i: collections.deque(maxlen=BUFFER_LIMIT) for i in range(NUM_AGENTS)}

num_episodes = 1000
for epi in range(num_episodes):
    obs = env.reset()
    s = flatten_obs(obs)
    score = {i: 0 for i in range(NUM_AGENTS)}
    
    for t in range(100): # 最大ステップ数
        actions = {i: agents[i].get_action(s[i]) for i in range(NUM_AGENTS)}
        next_obs, rewards, dones, info = env.step(actions)
        s_prime = flatten_obs(next_obs)

        for i in range(NUM_AGENTS):
            memory[i].append((s[i], actions[i], rewards[i], s_prime[i], any(dones.values())))
            score[i] += rewards[i]
        
        s = s_prime
        if any(dones.values()):
            break

        # 学習実行
        if len(memory[0]) > BATCH_SIZE:
            for i in range(NUM_AGENTS):
                batch = random.sample(memory[i], BATCH_SIZE)
                agents[i].train(batch)

    if epi % 50 == 0:
        print(f"Episode: {epi}, Average Score: {sum(score.values())/NUM_AGENTS:.2f}")

# --- 学習後の実行・可視化 ---
def run_learned_agent():
    obs = env.reset()
    s = flatten_obs(obs)
    for _ in range(20):
        display.clear_output(wait=True)
        env.render(mode='graphic')
        display.display(plt.gcf())
        time.sleep(0.5)

        actions = {i: agents[i].get_action(s[i]) for i in range(NUM_AGENTS)}
        next_obs, _, dones, _ = env.step(actions)
        s = flatten_obs(next_obs)
        if any(dones.values()): break

# run_learned_agent()