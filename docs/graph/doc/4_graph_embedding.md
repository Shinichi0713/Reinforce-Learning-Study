# グラフ埋め込み

## グラフ埋め込みとは

グラフ埋め込みとは、グラフの「頂点」や「辺」を**低次元のベクトル（数値の配列）に変換する技術**です。

### なぜ必要なのか

| 理由 | 説明 |
|------|------|
| 機械学習の入力にする | 機械学習モデルは数値ベクトルを受け取るため、グラフをそのまま入力できない |
| 類似度の計算 | ベクトル化すると「似ているノード」をコサイン類似度などで計算できる |
| 次元圧縮 | 100万ノードのグラフを100次元ベクトルに圧縮できる |
| 可視化 | 2次元や3次元に埋め込んで、グラフの構造を目で確認できる |

### 代表的な手法

| 手法 | 年 | 核心アイデア |
|------|:---:|:------------|
| DeepWalk | 2014 | ランダムウォークで「文」を生成し、Word2Vecでベクトル化 |
| node2vec | 2016 | DeepWalkを拡張。p, qパラメータでBFS/DFSバランスを調整 |
| LINE | 2015 | 1次近傍（直接のつながり）と2次近傍（共通の隣接ノード）を考慮 |
| GCN | 2016 | 畳み込みニューラルネットワークをグラフに拡張 |
| GraphSAGE | 2017 | 隣接ノードの特徴を集約してベクトルを生成。未学習ノードにも対応 |

### DeepWalkの仕組み（具体例）

**ステップ1：ランダムウォークで「文」を生成**

```
頂点Aから: A -> B -> A -> C -> A
頂点Bから: B -> A -> B -> A -> B
頂点Cから: C -> E -> C -> A -> B
...
```

**ステップ2：Skip-gramで学習**

「文の中で近い単語は似た意味を持つ」というWord2Vecの考え方をグラフに応用します。

**ステップ3：埋め込み結果**

```
各頂点の2次元ベクトル:
  A: [+0.5060, +0.7265]
  B: [+1.0228, +0.2635]
  C: [-0.4235, +0.9707]
  D: [+1.2152, -0.4838]
  E: [-0.5274, +1.8395]
  F: [+1.2955, -0.6900]
```

**コサイン類似度で「似ているノード」を発見：**

```
    A      B      C      D      E      F
A [1.000  +0.758  +0.524  +0.228  +0.631  +0.119]
B [+0.758  1.000  -0.159  +0.807  -0.027  +0.737]
C [+0.524  -0.159  1.000  -0.711  +0.991  -0.784]
...
```

- AとBの類似度が高い（0.758）→ 同じコミュニティ
- CとEの類似度が高い（0.991）→ 直接つながっている

### 可視化結果

ノード配置のイメージは以下のようなものとなります。

- **近いノード** = グラフ上で「似た位置」にあるノード
- **遠いノード** = グラフ上で「離れた位置」にあるノード

![1789073732340](image/4_graph_embedding/1789073732340.png)

### 確認の質問

グラフ埋め込みの手法の中で、**node2vec**はDeepWalkからどのような拡張を行っているでしょうか？（ヒント：2つのパラメータ `p` と `q` の役割について考えてみてください）


## ランダムウォーク

### 核心アイデア

グラフ上のある頂点から始めて、「隣接する頂点をランダムに選んで次々に移動する」ことで生成される頂点の系列です。

**直感的な例え:** 酔っ払った人が交差点でランダムに道を選んで歩く様子

### 定義

グラフ $G = (V, E)$ 上のランダムウォークとは、頂点の列 $v_0, v_1, v_2, \dots$ であって、各ステップで以下のルールに従うものです。

1. 時刻 $t$ に頂点 $v_t$ にいるとする
2. $v_t$ の隣接頂点（隣接する頂点全体）の中から、一様に（または重みに応じて）次の頂点 $v_{t+1}$ を選ぶ
3. 選ばれた頂点に移動する

単純な無向グラフの場合、頂点 $u$ から隣接頂点 $v$ への遷移確率は

$$P(u \to v) = \frac{1}{\deg(u)}$$

となります。ここで $\deg(u)$ は頂点 $u$ の次数（接続する辺の数）です。

### なぜ重要か

ランダムウォークは、グラフの構造を確率的に探索する最も基本的な方法であり、以下の性質が理論的・実用的に重要です。

**1. マルコフ連鎖としての性質**
ランダムウォークは**マルコフ連鎖**（次の状態が現在の状態だけに依存する確率過程）です。したがって、定常分布・混合時間・再帰性など、マルコフ連鎖の豊富な理論がそのまま適用できます。

**2. 定常分布（Stationary Distribution）**
長時間ランダムウォークを続けると、訪問する頂点の分布が一定に収束することがあります。無向グラフでは、この定常分布 $\pi(v)$ は

$$\pi(v) = \frac{\deg(v)}{2|E|}$$

となり、次数の高い頂点ほど頻繁に訪問されます。正則グラフ（全頂点の次数が等しいグラフ）では、定常分布は一様分布になります。

**3. 混合時間（Mixing Time）**
定常分布に「十分近づく」までに必要なステップ数です。グラフの連結性や「ボトルネック」の有無を測る指標として使われます。

**4. カバー時間（Cover Time）**
グラフの**すべての頂点を少なくとも1回訪問する**までにかかる期待ステップ数です。

### 代表的な応用

| 応用分野 | 具体例 |
|---------|--------|
| **PageRank** | Googleの検索ランキングアルゴリズム。ウェブページを頂点、リンクを辺とするグラフ上でランダムウォークを行い、訪問頻度でページの重要性を定量化 |
| **グラフ埋め込み** | DeepWalk、Node2Vec など。ランダムウォークで頂点の「文脈」を生成し、自然言語処理の技術で頂点をベクトル化 |
| **コミュニティ検出** | ランダムウォークが特定の頂点集合内に長く留まる傾向を利用し、クラスタ構造を発見 |
| **ネットワークサンプリング** | 巨大なソーシャルネットワークなどで、ランダムウォークを使って公平な標本を取得 |
| **機械学習** | グラフニューラルネットワーク（GNN）におけるメッセージ伝播の理論的基盤 |


### なぜグラフ埋め込みで使うのか

| 理由 | 説明 |
|------|------|
| 構造情報の保存 | よくつながっている頂点同士は系列の中で近くに現れやすい |
| コミュニティの反映 | 同じコミュニティの頂点は同じ系列に現れやすい |
| Word2Vecへの適用 | 系列を「自然言語の文」と見なしてSkip-gramを適用できる |

### Python実装

```python
import random

def random_walk(graph, start, length, seed=None):
    if seed is not None:
        random.seed(seed)

    walk = [start]
    current = start

    for _ in range(length - 1):
        neighbors = graph[current]
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
        current = next_node

    return walk
```

__実行結果__

```
始点 'A' から: A -> B -> A -> C -> A -> B -> A -> B -> A -> C
始点 'C' から: C -> E -> C -> A -> C -> A -> B -> A -> B -> A
始点 'E' から: E -> C -> A -> C -> A -> B -> A -> B -> A -> C
```


### node2vecとの関係

node2vecはこの「ランダムウォーク」を拡張し、**p（戻る確率）** と**q（探索バランス）** の2つのパラメータで歩き方を制御します。

| パラメータ | 小さい場合 | 大きい場合 |
|:---:|:---|:---|
| p | 来た頂点に戻りやすい（BFS的） | 戻りにくい（DFS的） |
| q | 外へ外へ進む（DFS的） | 近くを探索（BFS的） |

## Skip-Gram

### Word2Vec

「Word2Vec」とは、本来は自然言語処理（NLP）の手法ですが、**グラフの頂点をベクトル（数値列）に変換する「グラフ埋め込み（Graph Embedding）」の手法群で応用**されています。

グラフ埋め込みの文脈では、Word2Vecは以下のような流れで使われます。

**1. ランダムウォークで「文章」を生成する**
グラフ上でランダムウォークを行い、訪問した頂点の列を得ます。
例: $v_1 \to v_3 \to v_7 \to v_2 \to \dots$

**2. 頂点を「単語」、頂点列を「文章」とみなす**
NLPでは「単語の列＝文章」をWord2Vecに入力しますが、グラフでは「**頂点の列＝ランダムウォーク**」を入力します。

**3. Word2Vecで頂点をベクトル化する**
ランダムウォークで得られた頂点列から、**周囲の頂点と関係性を予測する学習**を行い、各頂点を低次元のベクトルに変換します。

この結果、以下の性質を持つベクトル表現が得られます。

- **近い頂点は近いベクトル**: ランダムウォークで頻繁に共起する頂点は、ベクトル空間上でも近くなる
- **構造的類似性の反映**: コミュニティ構造や役割の類似性がベクトルに現れる

このアプローチの代表的手法が **DeepWalk**（2014）や **Node2Vec**（2016）です。[arXiv - DeepWalk](https://ar5iv.labs.arxiv.org/html/1403.6652) [arXiv - node2vec](https://arxiv.org/pdf/1607.00653)

__例題:__

グラフ理論におけるランダムウォークをイメージするため例題を出題しました。

__問題設定__

カラテクラブグラフ（34頂点、2コミュニティ）を題材に、以下の流れでグラフ上のWord2Vec（DeepWalk風）を実装し、グラフのコミュニティ構造がベクトル表現に反映されることを確認する。

__Step 1: グラフの可視化（元の構造確認）__
- networkxのkarate_club_graph()を使用
- 34人のメンバーが2つのグループ（Mr. Hi / Officer）に分かれている実データ
- 頂点を色分けして元のコミュニティ構造を可視化する

__Step 2: ランダムウォークの生成__
- 各頂点から出発し、隣接頂点を一様ランダムに選んで移動する
- 各頂点から80回、長さ10のランダムウォークを生成
- 合計 34頂点 × 80回 = 2,720個のウォークを得る
- ウォークはWord2Vecへの「文章（文）」に相当する

__Step 3: Word2Vec（Skip-gram）による頂点のベクトル化__
- ランダムウォークをWord2Vecに入力
- パラメータ: sg=1（Skip-gram）、vector_size=64、window=5、epochs=50
- 各頂点が64次元の実数ベクトルとして表現される

__Step 4: 頂点間の類似度計算__
- 学習したベクトル間のコサイン類似度を計算
- 頂点0（Mr. Hiグループ）と類似する頂点は同グループか？
- 頂点33（Officerグループ）と類似する頂点は同グループか？
- 異なるコミュニティの頂点間の類似度は低いか？

__Step 5: t-SNEによる2次元可視化__
- 64次元のベクトルをt-SNEで2次元に圧縮
- 元のグラフ構造と、埋め込み空間での配置を並べて比較
- 同じコミュニティの頂点が埋め込み空間でも近くに集まるか確認する

__完全なPythonコード__

```python
"""
============================================================
グラフ理論 × Word2Vec（DeepWalk風）例題
============================================================
【必要ライブラリ】
  pip install networkx gensim matplotlib scikit-learn numpy
============================================================
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import random


# ------------------------------------------------------------
# Step 1: サンプルグラフの作成
# ------------------------------------------------------------
# カラテクラブグラフ: 34人のメンバーの交友関係
# 2つのグループ（Mr. Hi / Officer）に分かれている実データ
G = nx.karate_club_graph()

# ノードの色分け（Mr. Hi=赤、Officer=青）
node_colors = []
for n in G.nodes():
    if G.nodes[n]["club"] == "Mr. Hi":
        node_colors.append("#e74c3c")
    else:
        node_colors.append("#3498db")

# グラフの可視化
plt.figure(figsize=(8, 6))
plt.title("Karate Club Graph (Red=Mr. Hi, Blue=Officer)")
nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray",
        node_size=500, font_size=9)
plt.tight_layout()
plt.savefig("01_original_graph.png", dpi=150)
plt.show()

print(f"頂点数: {G.number_of_nodes()}, 辺数: {G.number_of_edges()}")


# ------------------------------------------------------------
# Step 2: ランダムウォークの実装
# ------------------------------------------------------------

def random_walk(graph, start_node, walk_length):
    """
    グラフ上でランダムウォークを行い、訪問した頂点のリストを返す。
    これがWord2Vecへの「文章（文）」に相当する。
    """
    walk = [start_node]
    current = start_node
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        walk.append(current)
    return walk


# 各頂点から出発するランダムウォークを複数生成
walks = []
num_walks_per_node = 80   # 各頂点から出発するウォーク数
walk_length = 10          # 1回のウォークで訪問する頂点数

random.seed(42)
np.random.seed(42)

for node in G.nodes():
    for _ in range(num_walks_per_node):
        walk = random_walk(G, node, walk_length)
        # Word2Vecは文字列のリストを受け取るので、頂点番号を文字列に変換
        walks.append([str(v) for v in walk])

print(f"生成したランダムウォークの総数: {len(walks)}")
print(f"例（最初のウォーク）: {walks[0]}")


# ------------------------------------------------------------
# Step 3: Word2Vec（Skip-gram）で頂点をベクトル化
# ------------------------------------------------------------
# パラメータ:
#   - sg=1: Skip-gramモデル（中心の頂点から周囲の頂点を予測）
#   - window=5: 周囲5個の頂点をコンテキストとして扱う
#   - vector_size=64: 各頂点を64次元ベクトルに変換
#   - min_count=1: 出現頻度1以上の頂点を対象

model = Word2Vec(
    sentences=walks,
    sg=1,               # Skip-gram
    vector_size=64,     # 埋め込み次元数
    window=5,           # コンテキストウィンドウサイズ
    min_count=1,
    workers=4,
    epochs=50,
    seed=42
)

# 頂点0のベクトル表現を確認
vec_node0 = model.wv["0"]
print(f"頂点0のベクトル（最初の10次元）: {vec_node0[:10]}")
print(f"ベクトル次数: {len(vec_node0)}")


# ------------------------------------------------------------
# Step 4: 頂点間の類似度を計算
# ------------------------------------------------------------
print("\n=== 頂点間の類似度（cosine similarity） ===")

# 頂点0と最も類似する頂点Top5
similar_to_0 = model.wv.most_similar("0", topn=5)
print("\n頂点0と最も類似する頂点:")
for node, score in similar_to_0:
    club = G.nodes[int(node)]["club"]
    print(f"  頂点{node:>2s} (グループ: {club:>10s}) -> 類似度: {score:.4f}")

# 頂点33と最も類似する頂点Top5
similar_to_33 = model.wv.most_similar("33", topn=5)
print("\n頂点33と最も類似する頂点:")
for node, score in similar_to_33:
    club = G.nodes[int(node)]["club"]
    print(f"  頂点{node:>2s} (グループ: {club:>10s}) -> 類似度: {score:.4f}")

# 2つの頂点間の類似度
sim_0_33 = model.wv.similarity("0", "33")
print(f"\n頂点0 と 頂点33 の類似度: {sim_0_33:.4f}")
print("（頂点0はMr. Hiグループ、頂点33はOfficerグループ → 異なるコミュニティなので類似度が低い）")


# ------------------------------------------------------------
# Step 5: t-SNEで2次元に圧縮して可視化
# ------------------------------------------------------------

# 全頂点のベクトルを取得
node_ids = [str(i) for i in G.nodes()]
embeddings = np.array([model.wv[nid] for nid in node_ids])

# t-SNEで2次元に圧縮
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

# プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左: 元のグラフ構造
ax1 = axes[0]
ax1.set_title("Original Graph Structure")
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors,
        edge_color="gray", node_size=500, font_size=9)

# 右: Word2Vec埋め込みの2次元可視化
ax2 = axes[1]
ax2.set_title("Word2Vec Embeddings (t-SNE 2D)")
for i, nid in enumerate(node_ids):
    x, y = embeddings_2d[i]
    color = node_colors[i]
    ax2.scatter(x, y, c=color, s=200, edgecolors="black", linewidths=0.5, zorder=3)
    ax2.annotate(nid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

# 凡例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", edgecolor="black", label="Mr. Hi"),
    Patch(facecolor="#3498db", edgecolor="black", label="Officer")
]
ax2.legend(handles=legend_elements)
ax2.set_xlabel("t-SNE Dimension 1")
ax2.set_ylabel("t-SNE Dimension 2")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_embedding_visualization.png", dpi=150)
plt.show()
```

__確認すべきポイント__

1. ランダムウォークで頻繁に共起する頂点ほど、Word2Vecのベクトルが近くなることを確認する
2. グラフのコミュニティ構造（Mr. Hi vs Officer）が、埋め込み空間上で自然に分離されることを確認する
3. Word2Vecがグラフの「構造的類似性」を数値ベクトルに変換できていることを理解する

__Word2Vecとグラフ理論の対応関係__

| 自然言語処理 | グラフ理論 |
|------------|-----------|
| 文章（単語の列） | ランダムウォーク（頂点の列） |
| 単語 | 頂点 |
| 単語の共起関係 | 頂点の隣接・近接関係 |
| 単語の分散表現 | 頂点の埋め込みベクトル |
| Skip-gram | 中心頂点から周囲頂点を予測 |

