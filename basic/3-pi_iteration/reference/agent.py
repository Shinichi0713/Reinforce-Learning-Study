import numpy as np
import matplotlib.pyplot as plt

class State:
    """状態を表すクラス"""
    def __init__(self, x, y, reward=0, is_goal=False):
        """
        x, y    : 現在位置
        reward  : 到達した時に得られる即時報酬
        is_goal : 迷路のゴール地点かどうか（ここに到達したら終了）
        """
        self.position = np.array([x, y], dtype=np.int64)
        self.reward = reward
        self.is_goal = is_goal

class Action:
    """行動を表すクラス"""
    def __init__(self, x, y):
        """x, y: 動く距離"""
        self.direction = np.array([x, y], dtype=np.int64)

class Environment:
    """環境を表すクラス"""
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
    def transition_at(self, s1, s2, a):
        """状態 s1 において行動 a を取ったときの状態 s2 への遷移確率"""
        # 移動先に通路があれば移動可能
        if np.all(s1.position + a.direction == s2.position):
            return 1.0
        else:
            return 0
    def reward(self, s1, s2):
        """状態 s1 から s2 へ遷移したときの即時報酬"""
        # 即時報酬は到達したマスで決まる
        return s2.reward

def draw_result(env, V, A):
    for s in env.states:
        i, j = s.position[0], s.position[1]
        if s.is_goal:
            plt.text(i-0.3, j, 'Goal')
        if s.reward != 0:
            plt.text(i-0.3, j-0.3, '$r={}$'.format(s.reward))
        if A[s] is not None:
            plt.scatter(i, j, color='black')
            plt.text(i+0.05, j, '{:.3f}'.format(V[s]))
            plt.quiver(i, j, A[s].direction[0], A[s].direction[1])
    x_max = max([s.position[0] for s in env.states])
    y_max = max([s.position[1] for s in env.states])
    v_grid = np.zeros([x_max+1, y_max+1])
    for s in env.states:
        i, j = s.position[0]-1-x_max, s.position[1]
        v_grid[i,j] = V[s]
    plt.imshow(v_grid.T, cmap='OrRd', origin='lower')
    plt.show()


class ValueBasePlanner:
    def __init__(self, env):
        self.env = env
        self.V_log, self.V, self.A = None, None, None
    def plan(self, gamma=0.8, eps=1e-5):
        """
        各状態 s ごとの価値 V[s] と、その状態からの最適な行動 A[s] を計算
        gamma : 割引率
        eps   : 収束判定の閾値
        """
        env = self.env
        states = env.states
        actions = env.actions
        V_log = {}  # 状態 s ごとの価値の更新ログ
        A = {}  # 状態 s における最適行動
        for s in states:
            V_log[s] = [0]
            A[s] = None
        i = 0
        while True:
            delta = 0
            for s in states:
                if s.is_goal:
                    V_log[s].append(V_log[s][-1])
                    continue
                v_max = -np.inf
                a_best = None
                for a in actions:
                    v = 0
                    for s_next in states:
                        t = env.transition_at(s, s_next, a)
                        v += t * (env.reward(s, s_next) + gamma * V_log[s_next][i])
                    if v > v_max:
                        v_max = v
                        a_best = a
                V_log[s].append(v_max)
                delta = max(delta, np.abs(V_log[s][i+1] - V_log[s][i]))
                A[s] = a_best
            if delta < eps:
                break
            i += 1
        self.V_log = V_log
        self.V = {s:V_log[s][-1] for s in states}
        self.A = A

# 問題のセットアップ
states = [
    State(0, 0), State(0, 1), State(0, 2), State(0, 3, 1.0, is_goal=True),
    State(1, 0), State(1, 2), State(1, 3, -1.0, is_goal=True),
    State(2, 0), State(2, 1), State(2, 2), State(2, 3)
]
actions = [Action(0, 1), Action(0, -1), Action(1, 0), Action(-1, 0)]
environment = Environment(states, actions)
# シミュレーション & 結果の描画
planner = ValueBasePlanner(environment)
planner.plan()
V, A = planner.V, planner.A
draw_result(environment, V, A)