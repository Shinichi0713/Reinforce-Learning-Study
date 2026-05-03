import numpy as np
import torch
import imageio
import os

# 環境とエージェントの初期化
env = TicTacToeEnv()
renderer = TicTacToeRenderer()

# 観測次元: 3x3=9, 行動次元: 9マス
obs_dim = 9
act_dim = 9

agent = DiscreteSACAgent(
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    auto_alpha=True,
)

# リプレイバッファ
buffer = ReplayBuffer(capacity=10000)

# 学習パラメータ
max_episodes = 1000
max_steps_per_episode = 10  # 五目並べは最大9手なので余裕を持って10
batch_size = 128
update_every = 1  # 何ステップごとに更新するか
save_every = 100  # 何エピソードごとにモデルを保存するか
render_every = 50  # 何エピソードごとに GIF を保存するか

os.makedirs("/tmp/models", exist_ok=True)
os.makedirs("/tmp/gifs", exist_ok=True)

# 学習ループ
for episode in range(max_episodes):
    state = env.reset()
    board = env.get_board_representation()
    obs = board.flatten()  # 観測: 3x3 -> 9次元ベクトル
    total_reward = 0.0
    frames = []

    for step in range(max_steps_per_episode):
        # 行動選択（観測から）
        action = agent.get_action(obs, deterministic=False)

        # 環境に適用
        next_state = env.step(action)
        next_board = env.get_board_representation()
        next_obs = next_board.flatten()

        # 報酬設計（プレイヤー0 の視点）
        if next_state.is_terminal():
            returns = next_state.returns()
            reward = returns[0]  # プレイヤー0 の報酬
            done = True
        else:
            reward = 0.0
            done = False

        total_reward += reward

        # 経験をバッファに保存
        buffer.push(obs, action, reward, next_obs, done)

        # 観測を更新
        obs = next_obs

        # レンダリング（一定間隔で）
        if episode % render_every == 0:
            frame = renderer.draw_board(board)
            frames.append(np.array(pygame.surfarray.array3d(frame).transpose(1, 0, 2)))

        if done:
            break

    # 一定ステップごとに学習
    if len(buffer) >= batch_size and episode % update_every == 0:
        for _ in range(1):  # 1エピソードあたり1回更新（必要に応じて増やす）
            batch = buffer.sample(batch_size)
            q_loss, policy_loss, alpha = agent.update(batch, batch_size)

    # モデル保存
    if episode % save_every == 0:
        agent.save_model(f"/tmp/models/sac_tic_tac_toe_{episode}.pth")

    # GIF 保存
    if episode % render_every == 0 and len(frames) > 0:
        imageio.mimsave(f"/tmp/gifs/episode_{episode}.gif", frames, fps=2)

    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
          f"Buffer Size: {len(buffer)}")

print("Training finished.")