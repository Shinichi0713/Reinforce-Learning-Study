import time
import numpy as np
from math import sqrt

from gym_env.rubiks_cube_env import RubiksCubeEnv

mcts_config = {
    'gamma': 0.99,
    'mu': 1.0E+00,
    'nu': 1.0E-01,
    'max_actions': 15,
    'unsolved_penalty': -10.0,
    'time_limit': 10,
    'max_runs': 1000
}


class MCTS(object):

    # コンストラクタ
    def __init__(self, agent):
        self.env = RubiksCubeEnv()
        self.agent = agent

        self.act_list = self.env.get_action_list()
        self.gamma = mcts_config['gamma']

        # exploration weight (c)
        self.mu = mcts_config['mu']

        self.max_actions = mcts_config['max_actions']
        self.unsolved_penalty = mcts_config[
            'unsolved_penalty']

        self.time_limit = mcts_config['time_limit']
        self.max_runs = mcts_config['max_runs']

    # 探索の遂行
    def run_search(self, sess, root_state):

        # --- PRE-PROCESS ---
        # 探索のルートノードの生成
        root_node = Node(None, None, None)

        # 最良経路の記録バッファ
        best_reward = float('-inf')
        best_solved = False
        best_actions = []

        # --- SEARCH MAIN ---
        n_run, n_done = 0, 0
        start_time = time.time()

        # 経路探索ループ
        while True:

            # 探索経路の初期化
            node = root_node
            state = root_state
            self.env.set_state(root_state)

            weighted_reward = 0.0
            done = [False]
            actions = []

            # 選択規則に沿った探索木の探索
            n_depth = 0
            while node.child_nodes:
                # 選択規則に沿った子ノードの選択
                node = self._select_next_node(
                    node.child_nodes)

                # 選択子ノード・探索ステップの評価
                next_state, reward, done, _ =\
                    self.env.step(node.action)
                weighted_reward += self.gamma**n_depth * \
                    reward[0]

                n_depth += 1
                state = next_state
                actions.append(node.action)

            # 既存探索木の子ノード展開
            if not done[0]:
                # 各行動の確率算出
                action_probs = self.agent.predict_policy(
                    sess, [state])
                # 各行動に対応する子ノードの生成
                node.child_nodes = [
                    Node(node, act, act_prob)
                    for act, act_prob in zip(
                        self.act_list,
                        action_probs[0])
                ]

            # 探索経路全体の評価
            if not done[0]:
                # utilize state value
                if 1:
                    # 終端ノードの評価
                    _v_s = self.agent.predict_value(
                        sess, [state])
                    weighted_reward +=\
                        self.gamma**n_depth * \
                        _v_s[0][0]

                    # 解けてない場合の罰則報酬
                    _penalty = self.unsolved_penalty
                    weighted_reward += _penalty

                # with random (not efficient) rollout policy
                if 0:
                    n_rollout_step = 0
                    state = None
                    done = [False]

                    while not done[0]:
                        action = np.random.choice(
                            self.act_list)
                        state, reward, done, _ = self.env.step(
                            action)
                        depth = n_depth + n_rollout_step
                        weighted_reward += self.gamma**depth * reward[
                            0]

                        n_rollout_step += 1
                        actions.append(action)

                        if len(actions) >= self.max_actions:
                            break

                    # add terminal state value if its still not done
                    if not done[0]:
                        _v_s = self.agent.predict_value(
                            sess, [state])
                        depth = n_depth + n_rollout_step
                        weighted_reward += self.gamma**(
                            depth) * _v_s[0][0]

            # 最終経路評価の経過ノードへの反映
            while node:
                node.n += 1
                node.v += weighted_reward
                node = node.parent_node

            # 最良経路の更新
            if best_reward < weighted_reward:
                best_reward = weighted_reward
                best_solved = done[0]
                best_actions = actions

            # 経路探索の終了条件
            n_run += 1
            if done[0]:
                n_done += 1
            duration = time.time() - start_time
            if n_run >= self.max_runs or duration >= self.time_limit:
                break

        # --- POST-PROCESS ---
        self.env.set_state(root_state)

        best_reward = 0.0
        best_solved = False
        best_states = [root_state]
        for i_act, action in enumerate(best_actions):
            next_state, reward, done, _ = self.env.step(
                action)
            best_reward += self.gamma**i_act * reward[0]
            best_states.append(next_state)
            if done[0]:
                best_solved = True
                break

        return best_reward, best_solved, best_states, best_actions

    # select next node based on a metric (ucb)
    def _select_next_node(self, child_nodes):
        metric = [
            self._calc_metric(node)
            for node in child_nodes
        ]
        best_node = child_nodes[np.argmax(metric)]
        return best_node

    # calc node evaluation (ucb) metric
    def _calc_metric(self, node):
        _u = self.mu * node.p * sqrt(
            node.parent_node.n) / (1.0 + node.n)
        # _q = node.v  # node.v - node.l
        _q = node.v / (1.0 + node.n)
        return _u + _q


class Node(object):

    def __init__(self, parent, action, prob):
        # 親ノードの登録
        self.parent_node = parent
        # 到達行動の登録
        self.action = action

        # 到達行動の選択確率の登録
        self.p = prob
        # ノードの状態価値
        self.v = 0.0
        # ノードの訪問回数
        self.n = 0

        # 展開子ノード
        self.child_nodes = []
