import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

class JaxCoopNavEnv:
    def __init__(self, size: int = 5):
        self.size = size
        self.num_agents = 2
        self.max_steps = 30
        self.agents = ["agent_0", "agent_1"]

        # ターゲット位置（両方同じターゲット）
        self.targets = jnp.array([[size-1, size-1], [size-1, size-1]])

        # ボトルネック（中央行の中央列のみ通行可能）
        self.bottleneck_row = size // 2
        self.bottleneck_cols = jnp.array([size // 2])

        # 行動空間（0: 停止, 1: 上, 2: 下, 3: 左, 4: 右）
        self.action_spaces = {agent: 5 for agent in self.agents}
        self.observation_spaces = {agent: 12 for agent in self.agents}  # 観測次元（後述）

    def reset(self, key: jax.Array) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        # エージェントの初期位置
        self.agent_pos = jnp.array([[0, 0], [self.size-1, self.size-1]])
        self.steps = 0
        obs = self._get_obs()
        return obs, self._get_state()

    def step(self, key: jax.Array, actions: Dict[str, int]) -> Tuple[
        Dict[str, jnp.ndarray], jnp.ndarray, Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        # actions: {"agent_0": int, "agent_1": int}
        actions_arr = jnp.array([actions["agent_0"], actions["agent_1"]])

        # 1. 行動を適用（JAX で書き直し）
        new_pos = self.agent_pos.copy()
        for i, a in enumerate(actions_arr):
            # 上
            new_pos = jnp.where(a == 1, new_pos.at[i, 0].set(jnp.maximum(0, new_pos[i, 0] - 1)), new_pos)
            # 下
            new_pos = jnp.where(a == 2, new_pos.at[i, 0].set(jnp.minimum(self.size-1, new_pos[i, 0] + 1)), new_pos)
            # 左
            new_pos = jnp.where(a == 3, new_pos.at[i, 1].set(jnp.maximum(0, new_pos[i, 1] - 1)), new_pos)
            # 右
            new_pos = jnp.where(a == 4, new_pos.at[i, 1].set(jnp.minimum(self.size-1, new_pos[i, 1] + 1)), new_pos)

        # 2. ボトルネック制約（中央行の特定列以外は通れない）
        for i in range(self.num_agents):
            r, c = new_pos[i]
            # 中央行かつボトルネック列以外なら元の位置に戻す
            cond = (r == self.bottleneck_row) & (~jnp.isin(c, self.bottleneck_cols))
            new_pos = jnp.where(cond, new_pos.at[i].set(self.agent_pos[i]), new_pos)

        # 3. 衝突チェック（同じセルには同時に入れない）
        collision = jnp.array_equal(new_pos[0], new_pos[1])
        new_pos = jnp.where(collision, self.agent_pos, new_pos)

        prev_pos = self.agent_pos.copy()
        self.agent_pos = new_pos

        # 4. 報酬計算（JAX で書き直し）
        rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        # ターゲットまでの距離
        dist0 = jnp.linalg.norm(self.agent_pos[0] - self.targets[0])
        dist1 = jnp.linalg.norm(self.agent_pos[1] - self.targets[1])
        global_dist_reward = -0.1 * (dist0 + dist1)
        rewards += global_dist_reward

        # 個別進捗報酬（距離が縮んだらボーナス）
        prev_dist0 = jnp.linalg.norm(prev_pos[0] - self.targets[0])
        prev_dist1 = jnp.linalg.norm(prev_pos[1] - self.targets[1])
        rewards = rewards.at[0].add(0.05 * (dist0 < prev_dist0))
        rewards = rewards.at[1].add(0.05 * (dist1 < prev_dist1))

        # 停滞ペナルティ（同じ位置に留まったらマイナス）
        rewards = rewards.at[0].add(-0.1 * jnp.array_equal(self.agent_pos[0], prev_pos[0]))
        rewards = rewards.at[1].add(-0.1 * jnp.array_equal(self.agent_pos[1], prev_pos[1]))

        # 同時到達ボーナス
        both_at_target = (dist0 == 0) & (dist1 == 0)
        rewards += 10.0 * both_at_target

        # 衝突ペナルティ
        collision = jnp.array_equal(self.agent_pos[0], self.agent_pos[1])
        rewards -= 1.0 * collision

        # ボトルネック付近の混雑ペナルティ
        bottleneck_pos = jnp.array([self.bottleneck_row, self.bottleneck_cols[0]])
        bottleneck_dist0 = jnp.linalg.norm(self.agent_pos[0] - bottleneck_pos)
        bottleneck_dist1 = jnp.linalg.norm(self.agent_pos[1] - bottleneck_pos)
        near_bottleneck = (bottleneck_dist0 < 2) & (bottleneck_dist1 < 2)
        rewards -= 0.5 * near_bottleneck

        # 協調報酬（相手がボトルネックに近いときに自分が遠いとボーナス）
        rewards = rewards.at[0].add(0.1 * ((bottleneck_dist1 < 1.5) & (dist0 > 1.0)))
        rewards = rewards.at[1].add(0.1 * ((bottleneck_dist0 < 1.5) & (dist1 > 1.0)))

        self.steps += 1
        done = (self.steps >= self.max_steps) | both_at_target

        # JaxMARL 形式の出力
        obs = self._get_obs()
        rewards_dict = {"agent_0": rewards[0], "agent_1": rewards[1]}
        done_dict = {"__all__": done, "agent_0": done, "agent_1": done}
        return obs, self._get_state(), rewards_dict, done_dict, {}

    def _get_obs(self) -> Dict[str, jnp.ndarray]:
        obs_list = []
        for i in range(self.num_agents):
            rel_dist = self.targets[i] - self.agent_pos[i]
            other_pos = self.agent_pos[1 - i] - self.agent_pos[i]
            bottleneck_rel = jnp.array([self.bottleneck_row, self.bottleneck_cols[0]]) - self.agent_pos[i]
            obs_i = jnp.concatenate([
                self.agent_pos[i] / self.size,
                rel_dist / self.size,
                other_pos / self.size,
                bottleneck_rel / self.size
            ])
            obs_list.append(obs_i)
        return {"agent_0": obs_list[0], "agent_1": obs_list[1]}

    def _get_state(self) -> jnp.ndarray:
        # グローバル状態（全エージェントの観測を結合）
        obs = self._get_obs()
        return jnp.concatenate([obs["agent_0"], obs["agent_1"]])

    @property
    def name(self) -> str:
        return "CoopNav"

    def observation_space(self, agent: str) -> int:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> int:
        return self.action_spaces[agent]