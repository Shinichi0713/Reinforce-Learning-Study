# 2. インポート
import numpy as np
import imageio
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# 3. 環境の作成（cramped_room レイアウト、協調タスク）
ae = AgentEvaluator.from_layout_name(
    mdp_params={"layout_name": "cramped_room", "old_dynamics": True},
    env_params={"horizon": 200}  # 1エピソードの長さ
)

# 4. ランダムエージェント同士のペアを作成
ap = AgentPair(RandomAgent(), RandomAgent())

# 5. 1エピソード分の軌跡を取得
trajs = ae.evaluate_agent_pair(ap, num_games=1)
traj = trajs["ep_states"][0]  # 最初のエピソードの状態列

# 6. 各状態を画像にレンダリングしてリストに保存
frames = []
visualizer = StateVisualizer()

for state in traj:
    # 状態を画像にレンダリング
    img = visualizer.render_state(state, grid=ae.env.mdp.grid)
    # numpy 配列に変換してフレームリストに追加
    frames.append(np.array(img))

# 7. フレームを MP4 動画として保存
imageio.mimsave("overcooked_demo.mp4", frames, fps=10)

print("Overcooked のデモ動画を overcooked_demo.mp4 として保存しました。")

