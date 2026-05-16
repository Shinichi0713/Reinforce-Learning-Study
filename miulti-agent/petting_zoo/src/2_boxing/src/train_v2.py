import torch
import torch.optim as optim
import os

import cv2


def estimate_boxer_positions_v2(obs_np):
    """
    obs_np: (210, 160, 3) の numpy 配列
    戻り値: (x1, y1), (x2, y2)
    """
    # 処理を軽くするためグレースケール化
    gray = cv2.cvtColor(obs_np, cv2.COLOR_RGB2GRAY)

    # 1P (白っぽい色: リングの床より明るい部分を抽出)
    mask1 = cv2.inRange(gray, 200, 255)
    
    # 2P (黒っぽい色: リングの床より暗い部分を抽出)
    mask2 = cv2.inRange(gray, 1, 50)

    def get_center(mask):
        M = cv2.moments(mask)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None

    pos1 = get_center(mask1)
    pos2 = get_center(mask2)
    
    return pos1, pos2

class MAPPOAtariTrainer:
    def __init__(self, env, agent, buffer_size=2048, batch_size=64, lr=3e-4, gamma=0.99, gae_lambda=0.95, ppo_epochs=10):
        self.env = env
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

        # ハイパーパラメータ
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # バッファの初期化
        # obs_shape=(4, 84, 84), joint_shape=(8, 84, 84)
        self.buffer = MAPPORolloutBuffer(buffer_size, (4, 84, 84), (8, 84, 84), self.device)

        # オプティマイザ (ActorとCriticをまとめて更新)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.stats = {"reward_history": [], "entropy_history": []}
        print(f"this trainer is run with {self.device}")

    def collect_rollouts(self):
        """データ収集（未完了エピソードの報酬も統計に含める）"""
        self.buffer.clear()
        obs_dict, _ = self.env.reset()

        # --- 報酬設計のパラメータ (前述の適応型) ---
        DISTANCE_REWARD_SCALE = 0.05
        CLOSE_PUNCH_BONUS = 0.1
        CLOSE_THRESHOLD = 20.0
        PUNCH_ACTIONS = {1, 10, 11, 12, 13, 14, 15, 16, 17}

        episode_rewards = []
        # 現在進行中のエピソード報酬を保持
        current_ep_reward = {'first_0': 0, 'second_0': 0}
        prev_dist = None

        for _ in range(self.buffer.buffer_size):
            o1, o2, joint_s = preprocess_joint_obs(obs_dict, self.device)
            
            with torch.no_grad():
                a1, logp1, _ = self.agent.get_action(o1.unsqueeze(0))
                a2, logp2, _ = self.agent.get_action(o2.unsqueeze(0))
                v1, v2 = self.agent.get_value(joint_s.unsqueeze(0))
            
            actions = {'first_0': a1.item(), 'second_0': a2.item()}
            next_obs_dict, rewards, terms, truncs, _ = self.env.step(actions)

            # --- 距離適応型報酬ロジック (画像推定) ---
            obs_raw = obs_dict['first_0']
            pos1, pos2 = estimate_boxer_positions_v2(obs_raw)
            
            shaping_reward = 0.0
            p_bonus_1p = 0.0
            p_bonus_2p = 0.0

            if pos1 and pos2:
                curr_dist = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
                if curr_dist > CLOSE_THRESHOLD:
                    if prev_dist is not None:
                        shaping_reward = (prev_dist - curr_dist) * DISTANCE_REWARD_SCALE
                else:
                    if a1.item() in PUNCH_ACTIONS: p_bonus_1p = CLOSE_PUNCH_BONUS
                    if a2.item() in PUNCH_ACTIONS: p_bonus_2p = CLOSE_PUNCH_BONUS
                prev_dist = curr_dist

            # 内部学習用報酬
            m_reward_1p = rewards['first_0'] + shaping_reward + p_bonus_1p
            m_reward_2p = rewards['second_0'] + shaping_reward + p_bonus_2p

            # 統計用（純粋なゲームスコアのみ蓄積）
            current_ep_reward['first_0'] += rewards['first_0']
            current_ep_reward['second_0'] += rewards['second_0']

            dones = [terms['first_0'] or truncs['first_0'], terms['second_0'] or truncs['second_0']]
            self.buffer.insert(
                o1, o2, joint_s, [a1.item(), a2.item()], [logp1.item(), logp2.item()],
                [m_reward_1p, m_reward_2p], [v1.item(), v2.item()], dones
            )

            obs_dict = next_obs_dict
            
            # エピソードが実際に終了した場合
            if any(dones):
                episode_rewards.append(current_ep_reward['first_0'])
                current_ep_reward = {'first_0': 0, 'second_0': 0}
                prev_dist = None
                obs_dict, _ = self.env.reset()

        # --- 【重要】バッファがいっぱいになった時点での報酬も記録する ---
        # これにより、長いエピソードの途中経過もログに反映されるようになります
        if not episode_rewards:
            episode_rewards.append(current_ep_reward['first_0'])

        # GAE計算
        _, _, last_joint_s = preprocess_joint_obs(obs_dict, self.device)
        with torch.no_grad():
            last_v1, last_v2 = self.agent.get_value(last_joint_s.unsqueeze(0))
        self.buffer.compute_returns_and_advantages(
            torch.tensor([last_v1.item(), last_v2.item()], device=self.device),
            self.gamma, self.gae_lambda
        )
        print(episode_rewards)
        return np.mean(episode_rewards) if episode_rewards else 0

    def train_step(self, clip_param=0.2, ent_coef=0.01, vf_coef=0.5):
        """学習を行い、平均エントロピーを返す（型変換を追加）"""
        entropies = []
        
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # --- 型変換：uint8 -> float32 への変換と正規化 ---
                obs = batch['obs'].view(-1, 4, 84, 84).float() / 255.0
                joint_states = batch['joint_states'].float() / 255.0
                # ----------------------------------------------
                
                actions = batch['actions'].view(-1)
                old_log_probs = batch['log_probs'].view(-1)
                advantages = batch['advantages'].view(-1)
                returns = batch['returns'].view(-1)

                # Actorの更新
                _, new_log_probs, dist_entropy = self.agent.get_action(obs, actions)
                entropies.append(dist_entropy.mean().item())

                # --- Loss計算 ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Criticの更新 (joint_states も float に変換したものを使用)
                v1_pred, v2_pred = self.agent.get_value(joint_states)
                v_preds = torch.cat([v1_pred, v2_pred], dim=0).squeeze()
                critic_loss = F.mse_loss(v_preds, returns)

                # トータルロス
                loss = actor_loss + vf_coef * critic_loss - ent_coef * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
        
        avg_entropy = np.mean(entropies)
        self.stats["entropy_history"].append(avg_entropy)
        return avg_entropy
    

    def save_checkpoint(self, iteration, path="checkpoints"):
        """モデルの重みを保存する"""
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.cpu()
        # 保存するデータの作成
        save_data = {
            'iteration': iteration,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # 1. 最新版として保存 (上書き用)
        torch.save(save_data, f"{path}/mappo_agent_latest.pth")
        
        # 2. 定期バックアップとして保存
        torch.save(save_data, f"{path}/mappo_agent_iter_{iteration}.pth")
        self.agent.to(self.device)
        print(f"Checkpoint saved at iteration {iteration}")

def train():
    # 全体の統合
    env = get_env()
    agent = MAPPOAgent(action_space_n=18)
    trainer = MAPPOAtariTrainer(env, agent)
    save_interval = 20  # 20イテレーションごとに保存
    
    # 学習ループ
    for iteration in range(100):
        # 1. データ収集時に平均報酬を取得
        avg_reward = trainer.collect_rollouts()
        
        # 2. 学習時に平均エントロピーを取得
        avg_entropy = trainer.train_step(ent_coef=0.1)

        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d} | Reward: {avg_reward:6.2f} | Entropy: {avg_entropy:.4f}")

        # パラメータの保存先を Google ドライブに変更
        if iteration > 0 and iteration % save_interval == 0:
            trainer.save_checkpoint(iteration, path=CHECKPOINT_DIR)