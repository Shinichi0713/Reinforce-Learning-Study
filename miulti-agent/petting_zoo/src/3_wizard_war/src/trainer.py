import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPPOTrainer:
    def __init__(self, 
                 actor_critic, 
                 device, 
                 ppo_epoch=10, 
                 num_mini_batch=4, 
                 clip_param=0.2, 
                 value_loss_coef=0.5, 
                 entropy_coef=0.01, 
                 max_grad_norm=0.5, 
                 huber_delta=10.0):
        self.ac = actor_critic
        self.device = device
        
        # ハイパーパラメータ
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta

        # 最適化（ActorとCriticを同時に更新）
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=3e-4, eps=1e-5)

    # def train(self, buffer):
    #     # アドバンテージを計算（バッファ内で正規化済みを想定）
    #     advantages = buffer.advantages
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #     train_info = {
    #         'value_loss': 0,
    #         'policy_loss': 0,
    #         'entropy': 0
    #     }

    def train(self, buffer):
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        train_info = {
            'value_loss': 0,
            'policy_loss': 0,
            'entropy': 0
        }

        for _ in range(self.ppo_epoch):
            data_generator = buffer.get_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                # flat_ids を sample から受け取る
                obs_b, state_b, actions_b, old_log_probs_b, \
                return_batch, adv_targ, masks_b, agent_ids_b = sample

                # evaluate_actions に agent_id_onehot を渡す
                values, action_log_probs, dist_entropy = self.ac.evaluate_actions(
                    obs_b, state_b, actions_b, agent_id_onehot=agent_ids_b
                )

                # 2. Policy Loss (L^CLIP)
                ratio = torch.exp(action_log_probs - old_log_probs_b)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                policy_loss = -torch.min(surr1, surr2).mean()

                # 3. Value Loss (L^VF) - Huber Lossを採用
                # 目標値との差分
                error = return_batch - values
                # value_loss = F.huber_loss(values, return_batch, delta=self.huber_delta)
                value_loss = F.huber_loss(values, return_batch, delta=50.0)

                # 4. Total Loss
                self.optimizer.zero_grad()
                (policy_loss - dist_entropy * self.entropy_coef + 
                value_loss * self.value_loss_coef).backward()
                
                # 勾配爆発の抑制
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ログ用
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['entropy'] += dist_entropy.item()

        return train_info