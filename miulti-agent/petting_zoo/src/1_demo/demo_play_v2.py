
import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 修正: seed 引数を削除
        return Discrete(3)  # ← ここを Discrete(3) に変更

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]
            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


# ===== 動作確認用コード =====

env_instance = env(render_mode="human")
env_instance.reset()

for agent in env_instance.agent_iter():
    observation, reward, termination, truncation, info = env_instance.last()

    if termination or truncation:
        action = None
    else:
        action = env_instance.action_space(agent).sample()

    env_instance.step(action)

env_instance.close()