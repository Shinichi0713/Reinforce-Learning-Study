import io
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_mappo_gif(env, trainer, filename="mappo_result.gif", max_steps=200):
    frames = []
    obs_list = env.reset()
    
    # --- GRUの初期隠れ状態をリセット ---
    # shape: (num_layers, batch=1, hidden_dim)
    h_actors = [torch.zeros(1, 1, trainer.hidden_act) for _ in range(trainer.num_agents)]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    print(f"🎬 エージェントの動作を録画中 (GRU対応版): {filename}")
    
    for t in range(max_steps):
        # --- 描画ロジック ---
        ax.clear()
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                ax.add_patch(patches.Rectangle((y, env.grid_size-1-x), 1, 1, fill=False, edgecolor='gray'))
        
        for pid, (pick, drop, picked, delivered) in enumerate(env.packages):
            px, py = (pick[1], env.grid_size - 1 - pick[0])
            dx, dy = (drop[1], env.grid_size - 1 - drop[0])
            if not picked: ax.add_patch(patches.Circle((px+0.5, py+0.5), 0.3, color="red"))
            if not delivered: ax.add_patch(patches.Circle((dx+0.5, dy+0.5), 0.3, color="green"))
        
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

        # 画像保存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))

        # --- エージェントの行動選択 (最新の正規化とGRUに対応) ---
        with torch.no_grad():
            # メソッド名を修正: _obs_to_tensor -> normalize_obs
            obs_tensor = trainer.normalize_obs(obs_list) 
            
            actions = []
            new_h_actors = []
            for i in range(env.num_agents):
                # 入力を (batch=1, seq=1, dim) に整形してGRUに渡す
                dist, h_a = trainer.actors[i](obs_tensor[i].view(1, 1, -1), h_actors[i])
                
                # 評価(GIF)時は、ランダムなsampleではなく、最も確率の高い行動(argmax)を選択
                a = torch.argmax(dist.probs)
                
                actions.append(a.item())
                new_h_actors.append(h_a)
            
            # 隠れ状態を更新
            h_actors = new_h_actors
        
        # 環境の更新
        obs_list, rewards, done, _ = env.step(actions)
        if done: break

    # GIF保存
    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"✅ 保存完了: {filename}")
    
    plt.close(fig)


save_mappo_gif(env, trainer)
