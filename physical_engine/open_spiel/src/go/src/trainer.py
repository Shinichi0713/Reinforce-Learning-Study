import os
import pyspiel
from open_spiel.python.algorithms.alpha_zero import alpha_zero

# 1. ゲームの設定
game_name = "go"
game_params = {"board_size": 9, "komi": 7.5}
game = pyspiel.load_game(game_name, game_params)

def train():
    # 2. Config の構築 (全必須引数を網羅)
    config_args = {
        "game": game_name,
        "path": "/tmp/alpha_zero_go",
        "nn_model": "resnet",
        "nn_width": 128,
        "nn_depth": 10,
        "train_batch_size": 128,
        "replay_buffer_size": 2**14,
        "replay_buffer_reuse": 4,
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "decouple_weight_decay": False,
        "checkpoint_freq": 100,
        "actors": 4,           # 並列自己対局数
        "evaluators": 1,       # 評価スレッド数
        "evaluation_window": 100,
        "eval_levels": 7,
        "uct_c": 2.0,
        "max_simulations": 400,
        "policy_alpha": 1.0,
        "policy_epsilon": 0.25,
        "temperature": 1.0,
        "temperature_drop": 20,
        "observation_shape": game.observation_tensor_shape(),
        "output_size": game.num_distinct_actions(),
        "max_steps": 10000,
        "quiet": False,
        "verbose": True
    }
    config = alpha_zero.Config(**config_args)

    print("Starting AlphaZero orchestrator for 9x9 Go...")
    
    # 3. 学習の実行
    # 手動でモデルや学習ループを作らず、オーケストレーター関数に Config を丸投げします。
    # この関数を実行するだけで、指定した max_steps (10000) まで自動で学習が進みます。
    alpha_zero.alpha_zero(config)
    
    print("Training finished!")

if __name__ == "__main__":
    save_path = "/tmp/alpha_zero_go"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train()