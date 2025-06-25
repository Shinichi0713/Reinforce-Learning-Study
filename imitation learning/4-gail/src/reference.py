import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np

# 環境を作成し、RolloutInfoWrapperでラップ
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
env = RolloutInfoWrapper(env)

# エキスパートデータを作成（ここではPPOで生成）
expert = PPO("MlpPolicy", env, verbose=1)

# RNGを作成
rng = np.random.default_rng()

# ロールアウトの取得
rollouts = rollout.rollout(
    expert, env, rollout.make_sample_until(min_timesteps=1000, min_episodes=None), rng
)
trajectories = [Trajectory(obs=path.obs, acts=path.acts, infos=path.infos, terminal=path.terminal) for path in rollouts]

# GAILの学習
learner = PPO("MlpPolicy", env, verbose=1)
gail_trainer = GAIL(
    demonstrations=trajectories,
    demo_batch_size=32,
    gen_algo=learner,
    n_disc_updates_per_round=4,
)
gail_trainer.train(total_timesteps=10000)

# 学習済みモデルのテスト
obs = env.reset()
for _ in range(1000):
    action, _ = learner.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
