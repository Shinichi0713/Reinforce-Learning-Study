**reco-gym は Google Colab でも動作可能です。**

Criteo Research が公開している強化学習用の推薦システム環境で、Colab 用の Getting Started ノートブックも用意されています。

---

## 1. Colab でのインストールと実行手順

### 1-1. reco-gym のインストール

Colab の新しいセルで以下を実行します。

```bash
!pip install recogym
```

これだけで、reco-gym 本体と依存ライブラリ（numpy, scipy, gym など）がインストールされます。

### 1-2. Getting Started ノートブックを動かす

GitHub リポジトリに Colab 用の入門ノートブックがあります。

- [Getting Started.ipynb](https://github.com/criteo-research/reco-gym/blob/master/Getting%20Started.ipynb)

Colab で開くには、以下のような手順がおすすめです。

1. 上記リンクを開く
2. 「Raw」ボタンを右クリックして URL をコピー
3. Colab で `!wget <URL>` でダウンロードし、`%run Getting\ Started.ipynb` で実行
   または、ノートブックの内容を Colab にコピペして実行

ノートブックには、環境の初期化、ランダム推薦エージェントの実行、報酬の集計などが含まれています。

---

## 2. reco-gym の特徴（Colab で動かす観点）

- **軽量な Python 環境**NumPy / SciPy / Gym ベースで、MuJoCo のような重い物理エンジンは不要です。
- **Colab 向けの Getting Started ノートブック**公式が Jupyter ノートブックを提供しており、Colab でもそのまま動かしやすいです。
- **推薦システム向けの RL 環境**
  - ユーザーの「有機的（organic）」行動と「バンディット（bandit）」行動をシミュレート
  - オフライン評価とオンライン強化学習の両方を扱える

---

## 3. まとめ

- reco-gym は **Colab でそのまま動かせる**環境です。
- `pip install recogym` でインストールし、公式の Getting Started ノートブックを Colab で実行すれば、すぐに推薦システムの強化学習を試せます。
- MuJoCo や複雑な物理エンジンは不要なので、AI Economist や multi-agent-emergence-environments よりもはるかに手軽です。

まずは Getting Started ノートブックを Colab で動かしてみるのがおすすめです。



**OpenSpiel は Google Colab で動作可能です。**  
DeepMind が公式に Colab 用のインストールノートブックとチュートリアルを用意しています。

---

## 1. Colab での動作可否とインストール

### 1-1. 公式の Colab サポート

OpenSpiel の GitHub リポジトリには、Colab 専用のノートブックが用意されています。

- インストール用ノートブック  
  [install_open_spiel.ipynb](https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/install_open_spiel.ipynb)
- チュートリアルノートブック  
  [OpenSpielTutorial.ipynb](https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/OpenSpielTutorial.ipynb)

README にも

> To try OpenSpiel in Google Colaboratory, please refer to open_spiel/colabs subdirectory or start here

と明記されており、**Colab での利用が公式に想定されています**。

### 1-2. インストール手順（Colab）

1. 上記のインストール用ノートブックを Colab で開く
2. セルを順に実行（C++ ビルドや Python バインディングのインストールが自動で行われる）
3. チュートリアルノートブックを開いて動作確認

Colab のランタイムは Ubuntu ベースなので、Linux 向けのインストール手順がそのまま使えます。

---

## 2. 可視化は可能か？

### 2-1. ゲーム盤面の描画

OpenSpiel は**ボードゲームやカードゲームの状態をテキストで表示する機能**を備えています。

例（Python）:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state)
```

これを Colab で実行すると、三目並べの盤面がテキスト（`x`, `o`, `.` など）で表示されます。

**グラフィカルな盤面描画（画像やアニメーション）は標準では提供されていません**が、  
Python 側で `print(state)` の結果を整形したり、matplotlib などを使って自分で描画することは可能です。

### 2-2. 学習ダイナミクス・評価指標の可視化

GitHub の説明によると、OpenSpiel には

> tools to analyze learning dynamics and other common evaluation metrics

（学習ダイナミクスや一般的な評価指標を分析するためのツール）

が含まれています。

具体的には、

- 学習曲線（報酬の推移）
- 勝率の推移
- 探索木の統計情報

などをプロットするためのユーティリティが提供されています。  
これらは matplotlib や seaborn などと組み合わせて、Colab 上でグラフとして描画できます。

---

## 3. まとめ

- **Colab での動作**：  
  → 公式のインストールノートブックとチュートリアルがあり、**問題なく動作します**。
- **可視化**：  
  - ゲーム盤面：テキスト表示は標準機能。グラフィカルな描画は自分で実装が必要。  
  - 学習結果：報酬・勝率などの可視化ツールが提供されており、Colab でグラフ化可能。

まずは公式のインストールノートブックとチュートリアルを Colab で実行してみるのがおすすめです。