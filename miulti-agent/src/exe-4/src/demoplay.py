import io
from PIL import Image

def save_mappo_gif(env, trainer, filename="mappo_result.gif", max_steps=100):
    frames = []
    obs_list = env.reset()
    
    # 保存用に新しいFigureを作成（既存の描画と干渉しないため）
    fig, ax = plt.subplots(figsize=(5, 5))
    
    print(f"🎬 エージェントの動作を録画中: {filename}")
    
    for t in range(max_steps):
        # --- 描画ロジック (renderメソッドの内容を流用) ---
        ax.clear()
        # グリッド
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                ax.add_patch(patches.Rectangle((y, env.grid_size-1-x), 1, 1, fill=False, edgecolor='gray'))
        # 荷物
        for pid, (pick, drop, picked, delivered) in enumerate(env.packages):
            px, py = (pick[1], env.grid_size - 1 - pick[0])
            dx, dy = (drop[1], env.grid_size - 1 - drop[0])
            if not picked: ax.add_patch(patches.Circle((px+0.5, py+0.5), 0.3, color="red"))
            if not delivered: ax.add_patch(patches.Circle((dx+0.5, dy+0.5), 0.3, color="green"))
        # エージェント
        colors = ["blue", "orange"]
        for i in range(env.num_agents):
            x, y = env.agent_pos[i]
            cx, cy = y, env.grid_size - 1 - x
            ax.add_patch(patches.Rectangle((cx, cy), 1, 1, color=colors[i], alpha=0.8))
            if env.agent_has[i] != -1:
                ax.text(cx+0.3, cy+0.3, "P", color="white", fontsize=12)
        
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect("equal")
        ax.set_title(f"Step: {t}")

        # --- 画像としてバッファに保存 ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))

        # --- エージェントの行動選択 ---
        with torch.no_grad():
            obs_tensor = trainer._obs_to_tensor(obs_list)
            actions = []
            for i in range(env.num_agents):
                # 決定論的な行動（最も確率が高い行動）を選択
                dist = trainer.actors[i](obs_tensor[i])
                actions.append(torch.argmax(dist.probs).item())
        
        # 環境の更新
        obs_list, rewards, done, _ = env.step(actions)
        
        if done:
            break

    # GIFの書き出し
    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=200, # 1コマ 0.2秒
            loop=0
        )
        print(f"✅ 保存完了: {filename}")
    
    plt.close(fig)

# 実行例
save_mappo_gif(env, trainer, "mappo_delivery_optimized.gif")