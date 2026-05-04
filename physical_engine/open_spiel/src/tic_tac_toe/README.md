
最近はMARL関係が多かったのですが、久しぶりに探索系のRLをしたくなりました。
環境構築が毎回手間だったのですが、そんな中で使えそうなOpen Spielについて調査、お試しを行いました。

## 概要
### 開発元
OpenSpielは **Google DeepMind**（現在は Google DeepMind の GitHub 組織として公開）が開発・公開している、ゲームにおける強化学習・探索・プランニングの研究用フレームワークです[OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)。


### サポートされているゲーム環境の種類

OpenSpielは、**70種類以上のゲーム環境**をサポートしており、以下のような多様なゲームタイプが含まれています[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)。

__1. シングルプレイヤー（1人用）ゲーム__
- **2048**: タイルをスライドさせて同じ数字を合体させ、2048タイルを作るパズル[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)  
- **Yacht**: サイコロを使った得点ゲーム（1〜10人対応）[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)  
- その他、単純なグリッドワールドやベンチマークタスクも含まれます。

__2. 2人・多人数の対戦ゲーム__
- **m,n,k-game**: m×n の盤面で k 個並べるゲーム（三目並べや五目並べの一般化）[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)  
- **Backgammon**: サイコロを使ったボードゲーム[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)  
- **Tic-Tac-Toe（三目並べ）**, **Go（囲碁）**, **Chess（チェス）** など、古典的な完全情報ゲーム  
- **Kuhn Poker**, **Leduc Poker** などの不完全情報ゲーム（ポーカー系）  
- その他、カードゲームやボードゲームの研究用ベンチマークが多数含まれます。

__3. 不完全情報ゲーム（情報が一部しか見えないゲーム）__
- **Kuhn Poker**: 3枚のカードを使う簡易ポーカー  
- **Leduc Poker**: もう少し複雑なポーカー  
- **Goofspiel**（スコアカードを使うカードゲーム）など  
- これらはゲーム理論や CFR（反実仮想後悔最小化）のベンチマークとしてよく使われます。

__4. 平均場ゲーム（Mean Field Games）__
- **Mean Field Game: linear-quadratic**: プレイヤーが一様分布からスタートし、同じ点に集まるように動く問題  
- **Mean Field Game: routing**: 各ノードで行き先を選ぶルーティング問題  
- 多数のエージェントが相互作用するようなマクロなモデルを扱うゲームです[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)。

__5. その他の特徴__
- **完全情報／不完全情報** の両方に対応  
- **同時手番（simultaneous-move）** と **交互手番（turn-taking）** の両方に対応  
- **ゼロサム／非ゼロサム（一般和）／協力ゲーム** など、報酬構造の異なるゲームを幅広くサポート[OpenSpiel Documentation](https://openspiel.readthedocs.io/en/latest/games.html)。

### ゲーム一覧の確認方法
公式ドキュメントの「Available games」ページに、ゲーム名・プレイヤー数・完全情報かどうか・利用可能かどうかなどが一覧で掲載されています[Available games — OpenSpiel documentation](https://openspiel.readthedocs.io/en/latest/games.html)。

- ページ内の表で、ゲーム名・プレイヤー数・情報構造（完全情報／不完全情報）・利用可否が確認できます。  
- また、GitHubリポジトリの `open_spiel/games/` ディレクトリに各ゲームの実装が格納されています[OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)。

### 特徴
豊富なアルゴリズム実装のようです。
OpenSpielには、強化学習・探索・ゲーム理論アルゴリズムのリファレンス実装が以下のジャンルに含まれています。

- 探索・プランニング系

MCTS（モンテカルロ木探索）
Minimax / Alpha-Beta など
- 強化学習系

DQN, A2C などの基本的な深層RL
マルチエージェントRL（NFSP, ED など）
- ゲーム理論・不完全情報ゲーム

CFR（反実仮想後悔最小化）
その派生アルゴリズム
線形計画法（LP）による均衡計算


## 使い方
### 1. インストール

__公式ドキュメント__
- メインリポジトリ: https://github.com/google-deepmind/open_spiel  
- 公式ドキュメント: https://openspiel.readthedocs.io/  
- インストール手順:
  - Linux / macOS: `docs/install.md`  
  - Windows: `docs/windows.md`  

__おおまかな手順（Linux/macOS 例）__
1. リポジトリをクローン
```bash
git clone https://github.com/google-deepmind/open_spiel.git
cd open_spiel
```

2. 依存パッケージのインストール（例: Ubuntu）
```bash
./open_spiel/scripts/install.sh
```

3. OpenSpielのビルド
```bash
mkdir build
cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) ../open_spiel
make -j$(nproc)
```

4. Pythonパッケージのインストール
```bash
pip install -e .
```

※詳細はOSごとに公式ドキュメントを参照してください。

### 2. Pythonでの基本的な使い方

__2.1 環境（ゲーム）の作成__

```python
import pyspiel

# ゲームの定義（例: じゃんけん）
game = pyspiel.load_game("matrix_rps")

# 状態の初期化
state = game.new_initial_state()
```

`load_game` の引数には、ゲームの名前（`kuhn_poker`, `leduc_poker`, `tic_tac_toe` など）を指定します。

__2.2 状態の情報を確認__

```python
print("ゲーム名:", game.get_type().short_name)
print("プレイヤー数:", game.num_players())
print("状態の文字列表現:", state)
print("現在のプレイヤー:", state.current_player())
print("可能な行動数:", state.legal_actions())
```

__2.3 行動を適用して進める__

```python
# 可能な行動の一覧
legal_actions = state.legal_actions()
print("可能な行動:", legal_actions)

# 最初の行動を適用
action = legal_actions[0]
state.apply_action(action)

print("行動後の状態:", state)
```

__2.4 終了状態と報酬__

```python
while not state.is_terminal():
    # ランダムに行動を選ぶ例
    action = np.random.choice(state.legal_actions())
    state.apply_action(action)

print("最終状態:", state)
print("報酬:", state.returns())
```

### 3. サンプルコードとチュートリアル

- `open_spiel/python/examples/` に多くのサンプルがあります。  
  - `example.py`: 基本的な使い方  
  - `mcts.py`: MCTS（モンテカルロ木探索）の例  
  - `jpsro.py`: ポリシー空間応答オラクル（PSRO）の例 など  
- `docs/concepts.md`: APIの概要と最初の例  
- Google Colab チュートリアル:  
  - https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/OpenSpielTutorial.ipynb  

### 4. 参考情報

- 公式GitHub: https://github.com/google-deepmind/open_spiel  
- 公式ドキュメント: https://openspiel.readthedocs.io/  
- チュートリアルColab: https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/OpenSpielTutorial.ipynb  


## お試し

五目並べを使って強化学習を実装してみました。

この手の手の探索はSACを使うのが著者の常習です。
本当はSACは状態空間が連続の場合に威力を発揮するといわれてますが、離散化も可能です。

コードは以下レポジトリを参考下さい。
https://github.com/Shinichi0713/Reinforce-Learning-Study/tree/main/physical_engine/open_spiel/src/tic_tac_toe

__結果__

今回はそこまでシビアな環境ではないので、あっさり学習出来ました。
最後に3回ランダムの打ち手と対戦しました。

```
=== Final Results ===
SAC Agent Wins: 3
Opponent Wins: 0
Draws: 0
Overall Winner: SAC Agent
```

<img src="image/README/marl_agent_motion.gif" alt="jssp-3" width="500px" height="auto">

## 総括
環境が初めからそろっていると強化学習のアルゴリズム実装に集中出来ます。
今回はそんな中からOpen Spielについて調査、お試しで実装してみました。

一点注意が可視化用の機能がないので、レンダリングする機能は自分で実装する必要があります。

