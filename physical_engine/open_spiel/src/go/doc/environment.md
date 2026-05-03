OpenSpielはゲームロジックとAPIを提供しますが、標準ではGUIや動画生成機能はありません。  
そのため、**UI表示と動画作成は外部ライブラリ（Pygameやmatplotlibなど）と組み合わせて実装**する必要があります。

以下では、**PygameでUI表示**と**matplotlibでアニメーション動画を作る**2つの代表的な方法を、Go（囲碁）を例に説明します。

---

## 1. PygameでUIを作り、対戦をリアルタイム表示する方法

### 1-1. 全体の流れ
1. OpenSpielでGoのゲームオブジェクトを作成
2. 自作のAI（ランダム、MCTSなど）を用意
3. Pygameで盤面と石を描画
4. 各ターンでAIが着手 → OpenSpielで状態更新 → Pygameで再描画
5. 必要ならスクリーンキャプチャで動画を録画

### 1-2. 実装のポイント

#### OpenSpiel側（Python）
```python
import pyspiel

# Go ゲームの作成
game = pyspiel.load_game("go", {"board_size": 9})  # 9路盤
state = game.new_initial_state()

# 例: ランダムAI（実際にはMCTSなどを使う）
def random_agent(state):
    return np.random.choice(state.legal_actions())

# 対戦ループ
while not state.is_terminal():
    action = random_agent(state)
    state.apply_action(action)
    # ここで盤面情報をPygameに渡す
```

#### Pygame側
- `state.observation_tensor()` や `state.board()` から盤面状態を取得
- 盤の線・交点・石を描画
- 毎ターン `pygame.display.flip()` で更新

#### 動画化
- OBS Studio などで画面キャプチャ → mp4保存
- もしくは `pygame.image.save()` でフレームを連番保存し、後でFFmpegで結合

---

## 2. matplotlibでアニメーション動画を作る方法

### 2-1. 手順
1. OpenSpielで対戦を最後までシミュレートし、**各ステップの盤面状態を保存**
2. `matplotlib.animation.FuncAnimation` で盤面をフレームごとに描画
3. `animation.save("go_game.mp4", writer="ffmpeg")` で動画保存

### 2-2. コード例（概要）

```python
import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

game = pyspiel.load_game("go", {"board_size": 9})
state = game.new_initial_state()

# 対戦を最後まで進め、各ステップの盤面を保存
boards = []
while not state.is_terminal():
    action = np.random.choice(state.legal_actions())
    state.apply_action(action)
    # 盤面を2D配列として取得（実装はOpenSpielのAPIに依存）
    board = state_to_board(state)  # 自作関数
    boards.append(board)

# アニメーション作成
fig, ax = plt.subplots()
im = ax.imshow(boards[0], cmap="gray", vmin=-1, vmax=1)

def update(frame):
    im.set_array(boards[frame])
    return [im]

ani = FuncAnimation(fig, update, frames=len(boards), blit=True)
ani.save("go_game.mp4", writer="ffmpeg", fps=2)
```

- `state_to_board()` は、OpenSpielの `state.observation_tensor()` や `state.board()` から盤面を2D配列に変換する関数です。
- `fps` を調整して再生速度を変えられます。

---

## 3. 既存のUIプロジェクトを参考にする

OpenSpiel公式のGitHub Discussionでは、PygameベースのUIプロジェクトが紹介されています[OpenSpiel GitHub Discussion](https://github.com/google-deepmind/open_spiel/discussions/1119)。

- **pygame_spiel**: OpenSpielゲームをPygameでGUI表示するプロジェクト  
  - Tic Tac ToeやBreakthroughなどが実装済み  
  - MCTS・DQNなどのAIと対戦可能  
  - ボード描画の実装パターンを参考に、Goへ拡張できます

このリポジトリの描画ロジックを参考に、Goの盤面・石を描画する部分を追加すれば、UI部分を一から作るより効率的です。

---

## 4. まとめ

- **UI表示**  
  - OpenSpielのPython APIでゲーム状態を取得  
  - Pygameで盤面・石を描画し、対戦をリアルタイム表示
- **動画作成**  
  - 対戦の全ステップを保存し、matplotlibのアニメーション機能でmp4生成  
  - もしくはPygame画面をOBS等で録画

OpenSpiel自体は「ゲームロジックとAPI」を提供し、**UIや動画は外部ライブラリと組み合わせて実装する**のが一般的なやり方です。  
Goの盤面描画は少し手間がかかりますが、既存のPygameプロジェクト（pygame_spiel）を参考にすると実装が楽になります[OpenSpiel GitHub Discussion](https://github.com/google-deepmind/open_spiel/discussions/1119)。