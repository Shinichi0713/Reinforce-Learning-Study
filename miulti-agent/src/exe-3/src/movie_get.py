device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = MultiSensorSearchEnv(size=GRID_SIZE, num_agents=NUM_AGENTS)
model = RoPETransformerAgent(
    grid_size=15, num_agents=3, action_space=5,
    num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_mult=FF_MULT,
).to(device)

assert os.path.exists(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth")), f"チェックポイントが見つかりません: {os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth")}"
state_dict = torch.load(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth"), map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("モデルのロードに成功しました。")

def run_episode(model, env, device, deterministic: bool = True, max_steps: int = MAX_STEPS, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    obs = env.reset()
    frames = [{
        "map": env.explored_map.copy(),
        "positions": dict(env.agent_positions),
        "step": 0,
        "coverage": float(np.sum(env.explored_map) / (env.size * env.size)),
    }]

    for step in range(1, max_steps + 1):
        with torch.no_grad():
            logits, _values = model(obs, device=device)
            dist = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()

        actions_dict = {i: actions[i].item() for i in range(env.num_agents)}
        obs, rewards, done, info = env.step(actions_dict)

        frames.append({
            "map": env.explored_map.copy(),
            "positions": dict(env.agent_positions),
            "step": step,
            "coverage": info["coverage"],
        })
        if done:
            break

    return frames


# deterministic=True: argmaxで行動選択(再現性重視) / False: 確率的サンプリング
# seed を指定すると初期配置を固定できます
frames = run_episode(model, env, device, deterministic=True, max_steps=MAX_STEPS, seed=None)

print(f"Total steps: {frames[-1]['step']} | Final coverage: {frames[-1]['coverage']*100:.1f}%")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

AGENT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def render_gif(frames, out_path: str, grid_size: int, num_agents: int, fps: int = 8):
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(idx):
        ax.clear()
        f = frames[idx]

        ax.imshow(
            f["map"], origin="lower", cmap="Greys", vmin=0, vmax=1,
            extent=(-0.5, grid_size - 0.5, -0.5, grid_size - 0.5), alpha=0.6,
        )

        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(num_agents):
            x, y = f["positions"][i]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.scatter(x, y, s=260, c=color, edgecolors="black", linewidths=1.2, zorder=3)
            ax.text(x, y, str(i), ha="center", va="center", color="white",
                     fontsize=10, fontweight="bold", zorder=4)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_title(f"Step {f['step']:3d} | Coverage: {f['coverage']*100:5.1f}%")
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 // fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)


OUT_PATH = "/content/rollout.gif"
render_gif(frames, OUT_PATH, grid_size=GRID_SIZE, num_agents=NUM_AGENTS, fps=8)
print("Saved:", OUT_PATH)

from IPython.display import Image, display
display(Image(filename=OUT_PATH))

