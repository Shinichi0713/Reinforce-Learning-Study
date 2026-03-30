QMIXは、**「個人のQ値が向上すれば、チーム全体のQ値も向上する（単調性制約）」**という前提のもと、離散的な行動空間で協力するタスクを得意とします。

---

## 実装問題：協調型ロボット倉庫の価値統合

### 1. 問題設定

2台のロボット（エージェント）が、共有された報酬（Shared Reward）を得るために倉庫内を移動しています。各エージェントは自分自身のネットワーク（Agent Network）を持ち、ローカルな観測から自身のQ値を出力します。

あなたは、これら2つの個別Q値を統合し、チーム全体のQ値を算出する **QMIXのMixing Networkクラス** をPyTorchで実装してください。

### 2. 仕様（制約）

* **入力:** 各エージェントのQ値（）と、環境のグローバルな状態ベクトル（）。
* **ハイパーネットワーク:** ミキシングネットワークの重み（Weights）は、を入力とする「ハイパーネットワーク」から生成してください。
* **単調性の保証:** QMIXのルールに従い、ミキシングネットワークの重みは**必ず正（非負）**でなければなりません。ハイパーネットワークの出力に `torch.abs()` または `F.softplus()` を適用してください。
* **バイアス:** バイアス項は正である必要はありません。
* **ネットワーク構造:**
* 隠れ層（Hidden dimension）のサイズは任意（例: 32）とします。
* 最終的に （スカラー値）を出力してください。



### 3. 実装テンプレート

以下のクラスの `forward` メソッドを完成させてください。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        # ハイパーネットワーク1 (重み w1 を生成)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * self.embed_dim)
        # ハイパーネットワーク2 (重み w2 を生成)
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim * 1)

        # バイアス項
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.v_head = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: 各エージェントのQ値 [batch_size, n_agents]
        states: グローバル状態ベクトル [batch_size, state_dim]
        """
        batch_size = agent_qs.size(0)
        
        # ここに実装を記述してください
        # 1. agent_qs を [batch_size, 1, n_agents] にリシェイプ
        # 2. states から w1, b1, w2, b2(v_head) を生成
        # 3. w1, w2 に絶対値を適用して正値を保証
        # 4. 行列演算を行い Q_tot を算出
        
        return q_tot

```

---

## ヒント：行列演算のコツ

QMIXの計算は、バッチ処理を行うために以下の行列積の形で行われます。

1. 第一層: $hidden = ELU(agent\_qs \times w1 + b1)$
2. 第二層: $Q_{tot} = (hidden \times w2) + v\_head(states)$

### 期待される出力の性質

この問題が正しく解けると、あるエージェントの  が増加した際に、 が減少することのない（単調増加または維持される）ネットワークが完成します。これがQMIXにおいて「分散実行（各エージェントが自分のQ値を最大化する行動を選ぶこと）」が「全体の利益最大化」に直結することを保証する鍵となります。
