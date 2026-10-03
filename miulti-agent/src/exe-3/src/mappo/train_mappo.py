# -------------------------------------------------------------------
# 3. 個人報酬対応 MAPPO 学習ルーチン (修正版)
# -------------------------------------------------------------------
PPO_EPOCHS = 4          # [FIX 4] 1エピソードあたりの更新epoch数
CLIP_RANGE = 0.2        # 一般的なPPOクリップ幅(0.8~1.2相当)に調整
GAMMA = 0.99
LAMBDA = 0.95
NUM_TRAIN = 1500

def train_mappo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiSensorSearchEnv(size=15, num_agents=3)
    model = RoPETransformerAgent(
        grid_size=15, num_agents=3, action_space=5,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_mult=FF_MULT,
    ).to(device)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth")):
        print("successfully load state dicts.")
        state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth"), map_location=device)
        model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(0, NUM_TRAIN):
        obs = env.reset()
        batch_obs, batch_actions, batch_logprobs, batch_rewards, batch_values = [], [], [], [], []
        info = {"coverage": 0.0, "terminated": False}

        for step in range(MAX_STEPS):
            batch_obs.append(obs)
            with torch.no_grad():
                logits, values = model(obs, device=device)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_dict = {i: actions[i].item() for i in range(env.num_agents)}
            next_obs, rewards_dict, done, info = env.step(actions_dict)

            # 各エージェント個別の報酬テンソル (3,)
            reward_tensor = torch.tensor([rewards_dict[i] for i in range(env.num_agents)], dtype=torch.float32, device=device)

            batch_actions.append(actions)
            batch_logprobs.append(log_probs)
            batch_rewards.append(reward_tensor)
            batch_values.append(values.squeeze(-1))

            obs = next_obs
            if done:
                break

        # [FIX 3] GAE計算: 真の終端(全マス探索完了)か打ち切り(MAX_STEP到達)かで
        # ブートストラップ値を切り替える。真の終端ではその先の価値は0。
        returns, advantages = [], []
        gae = torch.zeros(env.num_agents, device=device)

        with torch.no_grad():
            if info.get("terminated", False):
                next_val = torch.zeros(env.num_agents, device=device)
            else:
                _, next_value = model(obs, device=device)
                next_val = next_value.squeeze(-1)

        for t in reversed(range(len(batch_rewards))):
            nv = next_val if t == len(batch_rewards) - 1 else batch_values[t + 1]
            delta = batch_rewards[t] + GAMMA * nv - batch_values[t]
            gae = delta + GAMMA * LAMBDA * gae
            advantages.insert(0, gae.clone())
            returns.insert(0, gae + batch_values[t])

        b_actions = torch.stack(batch_actions)
        b_old_logprobs = torch.stack(batch_logprobs)
        b_returns = torch.stack(returns)
        b_adv = torch.stack(advantages)
        b_adv = (b_adv - b_adv.mean(dim=0, keepdim=True)) / (b_adv.std(dim=0, keepdim=True) + 1e-8)

        T = len(batch_obs)

        # [FIX 4] PPOパラメータ更新: 複数epoch回し、各epochで損失をT(タイムステップ数)
        # で正規化してから逆伝播する。旧実装は1エピソードにつき1回しか
        # optimizer.step()せず、かつ損失をTで平均していなかったため、
        # 長いエピソードほど勾配が肥大化し、更新も非常にサンプル非効率だった。
        for epoch in range(PPO_EPOCHS):
            optimizer.zero_grad()
            epoch_loss_sum = 0.0

            for t in range(T):
                logits, values = model(batch_obs[t], device=device)
                values = values.squeeze(-1)
                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(b_actions[t])
                ratio = torch.exp(new_logprobs - b_old_logprobs[t])

                surr1 = ratio * b_adv[t]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * b_adv[t]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns[t])
                entropy_bonus = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
                (loss / T).backward()
                epoch_loss_sum += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Total Step: {T:3d} | Coverage: {info['coverage']*100:.1f}%")
        if episode % 100 == 0:
            model.cpu()
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth"))
            model.to(device)

if __name__ == "__main__":
    train_mappo()