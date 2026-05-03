# Google Colab で pygame をヘッドレスで動かすための設定
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 画面表示を無効化（ヘッドレス）

import pygame
pygame.init()

import pygame
import numpy as np
import imageio
import os
from IPython.display import display, HTML

def play_vs_agent_and_record(agent_path, opponent_type="random", output_path="/tmp/tic_tac_toe_demo.mp4", fps=2):
    """
    1ゲーム分の対戦デモを録画する関数（1番勝負）
    """
    # 環境とエージェントの初期化
    env = TicTacToeEnv()
    renderer = TicTacToeRenderer(cell_size=100, margin=10)

    # エージェント（プレイヤー0）の読み込み
    agent = DiscreteSACAgent(obs_dim=9, act_dim=9)
    agent.load_model(agent_path)

    # pygame サーフェスの初期化（ウィンドウは表示しない）
    screen = pygame.Surface((renderer.width, renderer.height))

    # フレーム保存用リスト
    frames = []

    # ゲームループ
    state = env.reset()
    board = env.get_board_representation()
    obs = board.flatten()
    current_player = 0  # 0: SACエージェント, 1: 人間/ランダム

    game_over = False
    winner = None

    # 最初の盤面を保存
    frame = renderer.draw_board(board)
    frames.append(np.array(pygame.surfarray.array3d(frame).transpose(1, 0, 2)))

    while not game_over:
        if current_player == 0:
            # SAC エージェントのターン
            legal_actions = list(env.legal_actions())
            action = agent.get_action(obs, legal_actions=legal_actions, deterministic=True)
            state = env.step(action)
            board = env.get_board_representation()
            obs = board.flatten()
            current_player = 1
        else:
            # 人間 or ランダムのターン
            if opponent_type == "human":
                # Colab ではマウス入力が難しいため、ランダムで代用
                legal_actions = list(env.legal_actions())
                if legal_actions:
                    action = np.random.choice(legal_actions)
                    state = env.step(action)
                    board = env.get_board_representation()
                    obs = board.flatten()
                    current_player = 0
            elif opponent_type == "random":
                legal_actions = list(env.legal_actions())
                if legal_actions:
                    action = np.random.choice(legal_actions)
                    state = env.step(action)
                    board = env.get_board_representation()
                    obs = board.flatten()
                    current_player = 0

        # 盤面をフレームとして保存
        frame = renderer.draw_board(board)
        frames.append(np.array(pygame.surfarray.array3d(frame).transpose(1, 0, 2)))

        # ゲーム終了判定
        if env.is_terminal():
            game_over = True
            returns = state.returns()
            if returns[0] == 1:
                winner = "SAC Agent"
            elif returns[1] == 1:
                winner = "Opponent"
            else:
                winner = "Draw"

    # 動画として保存
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Demo saved to {output_path}")

    # Colab 上で動画を表示
    display(HTML(f"""
    <video width="400" controls>
      <source src="{output_path}" type="video/mp4">
    </video>
    """))

    return winner

def combine_videos(video_paths, output_path, fps=2):
    """
    複数の動画ファイルを結合して1つの mp4 に保存
    """
    frames = []
    for path in video_paths:
        reader = imageio.get_reader(path)
        for frame in reader:
            frames.append(frame)
        reader.close()

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Combined video saved to {output_path}")

def play_best_of_three_and_combine(agent_path, opponent_type="random", base_output_dir="/tmp", fps=2):
    """
    3番勝負を行い、各ゲームの動画を保存し、最後に結合する関数
    """
    os.makedirs(base_output_dir, exist_ok=True)

    results = []
    video_paths = []

    for game_idx in range(1, 4):
        output_path = os.path.join(base_output_dir, f"demo_{game_idx}.mp4")
        print(f"=== Game {game_idx} ===")
        winner = play_vs_agent_and_record(
            agent_path=agent_path,
            opponent_type=opponent_type,
            output_path=output_path,
            fps=fps
        )
        results.append(winner)
        video_paths.append(output_path)

    # 動画を結合
    combined_path = os.path.join(base_output_dir, "combined_demo.mp4")
    combine_videos(video_paths, combined_path, fps=fps)

    # 結果の集計
    sac_wins = results.count("SAC Agent")
    opponent_wins = results.count("Opponent")
    draws = results.count("Draw")

    print("\n=== Final Results ===")
    print(f"SAC Agent Wins: {sac_wins}")
    print(f"Opponent Wins: {opponent_wins}")
    print(f"Draws: {draws}")

    if sac_wins > opponent_wins:
        print("Overall Winner: SAC Agent")
    elif opponent_wins > sac_wins:
        print("Overall Winner: Opponent")
    else:
        print("Overall Result: Draw")

    # 結合後の動画を表示
    display(HTML(f"""
    <video width="400" controls>
      <source src="{combined_path}" type="video/mp4">
    </video>
    """))

    return results, video_paths, combined_path

# 使用例
if __name__ == "__main__":
    # 3番勝負（SAC vs Random）を実行し、動画を結合
    results, video_paths, combined_path = play_best_of_three_and_combine(
        agent_path="/tmp/models/sac_tic_tac_toe_900.pth",
        opponent_type="random",
        base_output_dir="/tmp",
        fps=2
    )

    # 各ゲームの動画パスと結合動画パスを表示
    print("\nSaved videos:")
    for path in video_paths:
        print(f"  {path}")
    print(f"Combined video: {combined_path}")