import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.atari import boxing_v2
import supersuit as ss
from ray.tune.registry import register_env

def env_creator(args):
    # 1. 環境の生成
    env = boxing_v2.parallel_env(render_mode="rgb_array")
    
    # 2. 前処理（順序が非常に重要です）
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.resize_v1(env, 84, 84)
    env = ss.color_reduction_v0(env, mode='full')
    env = ss.reshape_v0(env, (84, 84, 1))
    env = ss.dtype_v0(env, "float32") # ←【重要】ここを追加：入力をfloat型に変換
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # 0-1に正規化
    env = ss.frame_stack_v1(env, 4)
    
    return ParallelPettingZooEnv(env)

# Rayのクリーンな初期化
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=2)

register_env("boxing_mappo", lambda config: env_creator(config))

# 3. MAPPOの設定
config = (
    PPOConfig()
    .environment("boxing_mappo")
    .framework("torch")
    # 最新Rayでの型エラーと設定の混乱を避けるため、新しいAPIスタックを一旦オフにする
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .env_runners(num_env_runners=1)
    .training(
        gamma=0.99,
        lr=2.5e-4,
        model={
            "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]],
        }
    )
    .multi_agent(
        policies={"p0", "p1"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "p0" if "first" in agent_id else "p1",
    )
)

# 4. 学習の実行
tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"timesteps_total": 100000}),
    param_space=config.to_dict(),
)

print("データ型とAPIスタックを修正して学習を開始します...")
results = tuner.fit()
ray.shutdown()