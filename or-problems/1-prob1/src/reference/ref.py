# 環境の作成
periods = 30  # シミュレーションの日数
env_name = "InvManagement-v1" # 未達販売の後日販売を認めない場合の在庫管理を指定
env_config = {
    "periods": periods
}
env = create_env(env_name, env_config)

# 学習に用いるバラメータ
num_workers = 1
num_cpus_per_worker = 1
num_epochs_for_sgd_minibatch = 10
num_episodes_per_train = 10 * num_epochs_for_sgd_minibatch * num_workers

# 強化学習の設定
config = {
    "env": env,
    "framework": "torch",

    # 独立なエージェントの数と使用するCPU数
    "num_workers": num_workers,
    "num_cpus_per_worker": num_cpus_per_worker,

    # 学習率
    "lr_schedule": [[0, 5e-5], [400000, 1e-5], [800000, 5e-6], [1200000, 1e-6]],

    # ネットワークに関するパラメータ
    "vf_share_layers": False,
    "fcnet_hiddens": [64,64],
    "vfnet_hiddens": [64,64],
    "fcnet_activation": "relu",
    "vfnet_activation": "tanh",

    # 経験の収集に関するパラメータ
    # 参考：　https://docs.ray.io/en/latest/rllib/rllib-sample-collection.html
    "episode_mode": "complete_episodes", # workerから経験が収集されるとき、常にエピソード単位で収集される（エピソード途中までで終わっているrolloutは収集されない）
    "sgd_minibatch_size": periods * num_workers * num_epochs_for_sgd_minibatch, # sgdが実行される際のバッチサイズ
    "train_batch_size": periods * num_episodes_per_train, # 一回ポリシーを更新するにあたって収集されるバッチサイズ

    # GPUの数
    "num_gpus": 0
}

# 環境の登録
register_env(env_name, env_config)