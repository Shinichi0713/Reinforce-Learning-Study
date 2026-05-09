import torch
import torch.optim as optim
import os

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

        print(f"this trainer is run with {self.device}")

    def collect_rollouts(self):
        """環境を動かしてデータを収集する"""
        self.buffer.clear()
        obs_dict, _ = self.env.reset()

        for _ in range(self.buffer.buffer_size):
            # 1. 前処理
            o1, o2, joint_s = preprocess_joint_obs(obs_dict, self.device)

            # 2. 行動決定と価値予測
            with torch.no_grad():
                a1, logp1, _ = self.agent.get_action(o1.unsqueeze(0))
                a2, logp2, _ = self.agent.get_action(o2.unsqueeze(0))
                v1, v2 = self.agent.get_value(joint_s.unsqueeze(0))

            # 3. 環境の実行
            actions = {'first_0': a1.item(), 'second_0': a2.item()}
            next_obs_dict, rewards, terms, truncs, infos = self.env.step(actions)

            # 4. バッファへ保存
            dones = [terms['first_0'] or truncs['first_0'], terms['second_0'] or truncs['second_0']]
            self.buffer.insert(
                o1, o2, joint_s,
                [a1.item(), a2.item()],
                [logp1.item(), logp2.item()],
                [rewards['first_0'], rewards['second_0']],
                [v1.item(), v2.item()],
                dones
            )

            obs_dict = next_obs_dict
            if any(dones):
                obs_dict, _ = self.env.reset()

        # 5. GAEの計算準備 (最後の状態の価値)
        _, _, last_joint_s = preprocess_joint_obs(obs_dict, self.device)
        with torch.no_grad():
            last_v1, last_v2 = self.agent.get_value(last_joint_s.unsqueeze(0))

        self.buffer.compute_returns_and_advantages(
            torch.tensor([last_v1.item(), last_v2.item()], device=self.device),
            self.gamma, self.gae_lambda
        )

    def train_step(self, clip_param=0.2, ent_coef=0.01, vf_coef=0.5):
        """バッファのデータを使ってネットワークを更新する"""
        total_loss = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # 1Pと2Pのデータをまとめて処理するために変形
                # batch['obs']: (batch, 2, 4, 84, 84) -> (batch*2, 4, 84, 84)
                obs = batch['obs'].view(-1, 4, 84, 84)
                actions = batch['actions'].view(-1)
                old_log_probs = batch['log_probs'].view(-1)
                advantages = batch['advantages'].view(-1)
                returns = batch['returns'].view(-1)

                # 新しいログ確率とエントロピーを取得
                _, new_log_probs, dist_entropy = self.agent.get_action(obs, actions)

                # --- Actor Loss (PPO Clipping) ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic Loss (Value Function MSE) ---
                # 集中クリティックなので joint_states を使用
                v1_pred, v2_pred = self.agent.get_value(batch['joint_states'])
                # v_preds を (batch*2, 1) にまとめてリターンと比較
                v_preds = torch.cat([v1_pred, v2_pred], dim=0).squeeze()
                critic_loss = F.mse_loss(v_preds, returns)

                # --- Total Loss ---
                loss = actor_loss + vf_coef * critic_loss - ent_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5) # 勾配爆発防止
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss
    

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
        trainer.collect_rollouts() # 1. データを溜める
        loss = trainer.train_step() # 2. 学習する

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

        # パラメータの保存先を Google ドライブに変更
        if iteration > 0 and iteration % save_interval == 0:
            trainer.save_checkpoint(iteration, path=CHECKPOINT_DIR)

def load_latest_checkpoint(agent, optimizer, path=CHECKPOINT_DIR):
    latest_path = f"{path}/mappo_agent_latest.pth"
    
    if os.path.exists(latest_path):
        print(f"Checking for checkpoint at {latest_path}...")
        checkpoint = torch.load(latest_path)
        
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']
        
        print(f"Successfully loaded checkpoint. Resuming from iteration {start_iter}")
        return start_iter
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0