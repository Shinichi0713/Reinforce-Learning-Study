class SACAgent:
    def __init__(
        self,
        obs_dim,        # 観測次元（例: 15）
        act_dim,        # 行動次元（例: 2）
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        target_entropy=None,
        alpha_lr=3e-4,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # ネットワーク
        self.policy_net = PolicyNetwork(obs_dim, act_dim, hidden_dim)
        self.q_net1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q_net2 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_q_net1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.target_q_net2 = QNetwork(obs_dim, act_dim, hidden_dim)

        # ターゲットネットワークを初期化
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # オプティマイザ
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=lr)

        # エントロピー係数α
        self.alpha = alpha
        if auto_alpha:
            if target_entropy is None:
                target_entropy = -act_dim  # 一般的な設定
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

    def get_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.policy_net(obs_tensor)
                action = mu.squeeze(0).numpy()
            else:
                action, _ = self.policy_net.sample(obs_tensor)
                action = action.squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)

    def update(self, batch, batch_size):
        obs, act, rew, next_obs, done = batch

        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew).unsqueeze(-1)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done).unsqueeze(-1)

        # Q関数の更新
        with torch.no_grad():
            next_act, next_log_prob = self.policy_net.sample(next_obs)
            target_q1 = self.target_q_net1(next_obs, next_act)
            target_q2 = self.target_q_net2(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target = rew + (1 - done) * self.gamma * target_q

        # Q1, Q2の損失
        q1 = self.q_net1(obs, act)
        q2 = self.q_net2(obs, act)
        q_loss1 = F.mse_loss(q1, target)
        q_loss2 = F.mse_loss(q2, target)
        q_loss = q_loss1 + q_loss2

        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_loss.backward()
        # 勾配クリッピングを追加
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q_optimizer1.step()
        self.q_optimizer2.step()

        # 方策の更新
        new_act, new_log_prob = self.policy_net.sample(obs)
        q1_new = self.q_net1(obs, new_act)
        q2_new = self.q_net2(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob.unsqueeze(-1) - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # 勾配クリッピングを追加
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # αの自動調整（オプション）
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # αの勾配もクリップ（任意）
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ターゲットネットワークのソフトアップデート
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item(), self.alpha


# ====================== 経験再生バッファ ======================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        return obs, act, rew, next_obs, done

    def __len__(self):
        return len(self.buffer)


# ====================== 学習ループ ======================
def train_sac_on_robot_carry():
    # 環境とエージェントの設定
    env = RobotCarryEnv(max_steps=200, world_size=10.0)
    obs_dim = env.observation_space.shape[0]  # 15
    act_dim = env.action_space.shape[0]      # 2

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        target_entropy=-act_dim,
        alpha_lr=3e-4,
    )

    # 経験再生バッファ
    buffer = ReplayBuffer(capacity=100000)
    batch_size = 256
    start_steps = 1000  # ランダム行動でバッファをためるステップ数

    # 学習パラメータ
    max_episodes = 1000
    update_interval = 1  # 1ステップごとに更新
    eval_interval = 50     # 50エピソードごとに評価
    save_gif_interval = 200  # 200エピソードごとにGIF保存

    # 学習ループ
    total_steps = 0
    for episode in range(max_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        # エピソードループ
        while not (done or truncated):
            # 行動選択（初期はランダム）
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs, deterministic=False)

            # 環境ステップ
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # バッファに保存
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_steps += 1

            # 一定ステップ以上たまったら学習
            if len(buffer) >= batch_size and total_steps >= start_steps:
                if total_steps % update_interval == 0:
                    batch_obs, batch_act, batch_rew, batch_next_obs, batch_done = buffer.sample(batch_size)
                    batch = (batch_obs, batch_act, batch_rew, batch_next_obs, batch_done)
                    q_loss, policy_loss, alpha = agent.update(batch, batch_size)

        # エピソード終了時のログ
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Steps {total_steps}, Reward: {episode_reward:.2f}")

        # 評価（deterministicな方策でテスト）
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, num_episodes=5)
            print(f"[EVAL] Episode {episode}, Eval Reward: {eval_reward:.2f}")

        # GIF保存（学習の様子を可視化）
        if episode % save_gif_interval == 0:
            record_episode_as_gif(env, agent, f"episode_{episode}.gif")

    # 学習終了後、最終方策でGIFを保存
    record_episode_as_gif(env, agent, "final_policy.gif")


def evaluate_agent(env, agent, num_episodes=5):
    """評価用：deterministicな方策で複数エピソードを実行し、平均報酬を返す"""
    total_reward = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.get_action(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

def record_episode_as_gif(env, agent, filename, max_steps=200):
    """1エピソードをGIFとして保存"""
    env.start_recording()
    obs, _ = env.reset()
    done = False
    truncated = False
    step_count = 0
    while not (done or truncated) and step_count < max_steps:
        action = agent.get_action(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step_with_record(action)
        step_count += 1
    env.stop_recording()
    env.save_gif(filename)