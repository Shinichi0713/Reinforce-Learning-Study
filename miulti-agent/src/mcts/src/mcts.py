import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class MCTSNode:
    """MCTSのノード（状態）を表すクラス"""
    def __init__(self, state_id, parent=None, action=None):
        self.state_id = state_id      # 状態のID（表示用）
        self.parent = parent          # 親ノード
        self.action = action          # 親からこのノードに至る行動
        self.children = []            # 子ノードのリスト
        self.visits = 0               # 訪問回数
        self.total_reward = 0.0       # 累積報酬（勝った回数など）
        self.q_value = 0.0            # Q値 = total_reward / visits

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        # ここでは簡易的に「状態IDが5以上なら終端」とします
        return self.state_id >= 5

    def ucb_score(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')  # 未訪問なら最大値
        # UCB1: Q + C * sqrt(ln(parent_visits) / visits)
        return self.q_value + exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda c: c.ucb_score(exploration_weight))

    def expand(self):
        # 簡易的に「子ノードを1つずつ追加」する例
        if self.is_terminal():
            return self
        new_state_id = self.state_id + 1
        child = MCTSNode(new_state_id, parent=self, action=f"a{new_state_id}")
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward
        self.q_value = self.total_reward / self.visits


def simulate_random_playout(node):
    """ランダムプレイアウト（簡易版）"""
    # ここでは「状態IDが偶数なら報酬1、奇数なら0」とします
    return 1.0 if node.state_id % 2 == 0 else 0.0


def build_tree_graph(root):
    """networkxで木構造を構築"""
    G = nx.DiGraph()
    node_queue = [root]
    while node_queue:
        node = node_queue.pop(0)
        G.add_node(node.state_id, visits=node.visits, q=node.q_value)
        for child in node.children:
            G.add_edge(node.state_id, child.state_id, action=child.action)
            node_queue.append(child)
    return G


def draw_mcts_tree(root, step, selected_path=None):
    """MCTSの木を描画（訪問回数とQ値を表示）"""
    G = build_tree_graph(root)
    pos = nx.spring_layout(G, seed=42)  # レイアウトを固定

    plt.figure(figsize=(10, 6))
    plt.title(f"MCTS Step {step}\n"
              f"Selected path: {selected_path if selected_path else 'None'}")

    # ノードの色: 訪問回数が多いほど濃い青
    node_colors = [d['visits'] for _, d in G.nodes(data=True)]
    # 修正箇所: キーは node_id だけにする
    node_labels = {node_id: f"S{node_id}\nQ={d['q']:.2f}\nV={d['visits']}" 
                   for node_id, d in G.nodes(data=True)}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Blues,
                           node_size=1500, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 選択されたパスを強調表示
    if selected_path:
        path_edges = list(zip(selected_path[:-1], selected_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               edge_color='red', width=3, arrows=True)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def run_mcts_with_visualization(num_steps=10):
    """MCTSを実行し、各ステップで木を可視化"""
    root = MCTSNode(state_id=0)
    selected_path_history = []

    for step in range(num_steps):
        # 1. Selection: Q値（UCB）に基づいてノードを選択
        node = root
        path = [node.state_id]
        while not node.is_leaf():
            node = node.best_child(exploration_weight=1.0)
            path.append(node.state_id)

        # 2. Expansion: 未展開なら展開
        if not node.is_terminal():
            node = node.expand()
            path.append(node.state_id)

        # 3. Simulation: ランダムプレイアウト
        reward = simulate_random_playout(node)

        # 4. Backpropagation: Q値を更新
        backprop_node = node
        while backprop_node is not None:
            backprop_node.update(reward)
            backprop_node = backprop_node.parent

        selected_path_history.append(path)

        # 5. このステップでの木の状態を可視化
        draw_mcts_tree(root, step=step+1, selected_path=path)

    return root, selected_path_history


if __name__ == "__main__":
    root, paths = run_mcts_with_visualization(num_steps=8)