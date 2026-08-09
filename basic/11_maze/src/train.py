"""
報酬の抜け穴(regress_penalty)修正後に学習を実行し、
ゴール到達率がエピソードを通じてどれだけ「安定」しているかを診断するスクリプト。
 
見たいもの:
  - 学習後半（収束したはずの区間）で、ウィンドウごとの成功率のばらつき（分散・標準偏差）
  - ばらつきが小さければ「安定して同程度の成功率が出ている」
  - ばらつきが大きければ、まだ学習・報酬設計・カリキュラム側に改善余地がある
"""
 
import statistics
import json
 
 
def windowed_success_rates(success_history, window=50):
    """success_historyを固定幅windowで区切り、各ウィンドウの成功率のリストを返す"""
    rates = []
    for i in range(0, len(success_history) - window + 1, window):
        chunk = success_history[i:i + window]
        rates.append(sum(chunk) / len(chunk))
    return rates
 
 
def summarize(name, rates):
    if len(rates) < 2:
        print(f"[{name}] ウィンドウ数が少なすぎて分散を計算できません（{len(rates)}個）")
        return
    mean = statistics.mean(rates)
    stdev = statistics.stdev(rates)
    cv = stdev / mean if mean > 0 else float("inf")  # 変動係数（相対的なばらつき）
    print(f"[{name}]")
    print(f"  ウィンドウ数: {len(rates)}")
    print(f"  各ウィンドウの成功率: {[f'{r:.0%}' for r in rates]}")
    print(f"  平均成功率: {mean:.1%}")
    print(f"  標準偏差: {stdev:.3f}")
    print(f"  変動係数(std/mean): {cv:.2f}  "
          f"(目安: 0.2以下なら概ね安定、0.4以上ならまだ不安定)")
    print()
    return {"mean": mean, "stdev": stdev, "cv": cv, "rates": rates}
 
 
num_episodes = 1000        # 数百〜千エピソード。必要に応じて調整してください
window = 50                 # 何エピソードごとに成功率をまとめて見るか
max_steps_per_episode = 60  # 5x5迷路なら十分な余裕を持った値

env = MazeEnv(maze_file="nonexistent_maze_file.txt", rows=5, cols=5)

agent = TransformerPPOAgent(
    env,
    num_layers=2,          # まずは現行構成のまま。表現力を試すときはここを3,4に
    entropy_coef=0.05,
    entropy_coef_final=0.005,
    path_save=path_save,
)

episode_rewards, success_history = agent.train(
    num_episodes=num_episodes,
    max_steps_per_episode=max_steps_per_episode,
    log_interval=max(1, num_episodes // 20),
    use_curriculum=True,
    curriculum_start_distance=3,
    curriculum_success_threshold=0.7,
    curriculum_window=20,
)

print("\n" + "=" * 60)
print("学習完了。安定性を分析します。")
print("=" * 60 + "\n")

# 全区間
all_rates = windowed_success_rates(success_history, window=window)
summarize("全エピソード", all_rates)

# 後半のみ（カリキュラムが最大難易度に達し、収束していると期待される区間）
half = len(success_history) // 2
latter_rates = windowed_success_rates(success_history[half:], window=window)
summarize("後半エピソードのみ（収束後の安定性の目安）", latter_rates)

# 結果をファイルに保存（次回以降の比較用）
result = {
    "num_episodes": num_episodes,
    "window": window,
    "success_history": success_history,
    "episode_rewards": episode_rewards,
}
with open("/content/stability_check_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("結果を /content/stability_check_result.json に保存しました。")
 
