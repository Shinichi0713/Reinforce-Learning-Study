
先日まで取り組んだwizard warやボクシング以外にPettingZooで使えるマルチエージェント環境がないかを調査しました。

## 概要
PettingZooには、wizard warやボクシング以外にも**協調（cooperative）を前提としたマルチエージェント環境が複数存在します**。

代表的なものとして、**SISLカテゴリ**に含まれる以下の環境が挙げられます。

### SISL環境（協調マルチエージェントベンチマーク）

PettingZoo公式ドキュメントでは、SISL環境について

> The SISL environments are a set of three cooperative multi-agent benchmark environments, created at SISL (Stanford Intelligent Systems Laboratory)…  
> （SISL環境は、SISLで作成された3つの協調マルチエージェントベンチマーク環境のセットです）

と明記されており、**協調型マルチエージェント環境として設計されています**[PettingZoo SISL Docs](https://pettingzoo.farama.org/environments/sisl)。

このカテゴリに含まれる協調型環境の例としては、少なくとも以下が確認できます。

- **Multiwalker**  
  複数の歩行ロボット（walker）が協力して前進するタスクです。個々のエージェントが単独で進むのは難しく、**全体として歩行を維持・前進させるために協調が必要**な設計になっています[PettingZoo Multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker)。

- **Pursuit**  
  複数の追跡エージェントが協力して、逃げるターゲットを捕まえるタスクです。個々のエージェントが単独でターゲットを捕まえるのは難しく、**包囲・連携によって初めて効率的に捕獲できる**ような協調構造になっています[PettingZoo Pursuit](https://pettingzoo.farama.org/environments/sisl/pursuit)。

これらはいずれも、**「全エージェントの合計報酬を最大化する」という共通の目標**を持ち、**エージェント間の競争ではなく協力が前提**になっている点で、wizard warやボクシング（対戦型）とは明確に異なります。

### その他の協調型環境の可能性

PettingZooには、SISL以外にもさまざまな環境カテゴリ（Atari、Third-Party Environments など）が含まれています[PettingZoo Documentation](https://pettingzoo.farama.org/index.html)。  
それらの一部には、**元々のゲームが協調プレイを想定しているもの**（例：協力型シューティング、協力型パズルなど）をマルチエージェント化した環境が含まれている可能性があります。

ただし、公式ドキュメント上で「cooperative」と明示されている環境は、主に**SISLカテゴリ**が中心です。  
もし「wizard warやボクシング以外で、明示的に協調型とされている環境を知りたい」という意図であれば、**Multiwalker**と**Pursuit**が最も代表的な例になります。

### まとめ

- PettingZooには、wizard warやボクシング（対戦型）とは別に、**協調型マルチエージェント環境が存在します**。
- その代表例は **SISLカテゴリの Multiwalker と Pursuit** で、いずれも「複数エージェントが協力して共通の目標を達成する」設計になっています[PettingZoo SISL Docs](https://pettingzoo.farama.org/environments/sisl)。
- その他のカテゴリにも協調的なタスクをマルチエージェント化した環境が含まれる可能性がありますが、公式に「cooperative」と明示されているのは主にSISL環境です。

前回までの反省点で、取り組んだゲームはお互いでチームプレーするということがありがたがられないようなゲームでした。(wizard warもスタンドプレーのベストパフォーマンスを行えばハイスコアを狙える)
仕様を調べると Persuit はチームプレーが必要とされそうで、MARLらしいというように思いました。

## Persuitの仕様

Pursuit環境では、**プレーヤは「追跡者（pursuer）」側のエージェント**で、**逃げる側（evader）はランダムに動く**という設定になっています。

### プレーヤ（エージェント）の役割

- **Pursuer（追跡者）**  
  - プレーヤが制御するエージェントです。  
  - デフォルトでは **8体の赤いエージェント**として登場します。  
  - グリッド上を移動し、逃げるエージェントを**包囲して捕まえる**ことが役割です。

- **Evader（逃げる側）**  
  - プレーヤは直接操作しません。  
  - デフォルトでは **30体の青いエージェント**として登場します。  
  - **ランダムに動く**設定になっています（プレーヤが制御するのはpursuerのみ）[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

### ゲームの目的と報酬

ゲームの目的は、**追跡者（pursuer）が協力して、できるだけ多くの逃げるエージェント（evader）を捕まえること**です。

- **捕獲（完全包囲）したとき**  
  - 逃げるエージェントを**周囲から完全に囲んだとき**、その周りにいる追跡者それぞれに **+5.0** の報酬が与えられます。  
  - 捕まえられたエージェントは環境から取り除かれます。

- **タッチ（接触）したとき**  
  - 追跡者が逃げるエージェントに**触れた（タグした）とき**、その追跡者に **+0.01** の報酬が与えられます。

- **ステップごとの「緊急度」ペナルティ**  
  - 各ステップで、追跡者に **-0.1** の「urgency reward」が適用されます。  
  - これは、**ダラダラ時間をかけるよりも素早く捕まえる方が有利**になるように設計されたペナルティです[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

### ゲームの終了条件

- **すべての逃げるエージェントを捕まえたとき**  
  → ゲームクリア（成功）として終了します。

- **500サイクル経過したとき**  
  → タイムリミットに達し、ゲーム終了となります[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

## 環境とのやり取り
前回までの反省点を踏まえてゲーム始める前に環境について理解したいところです。
Pursuit環境からは、各エージェントに対して以下の情報が得られます。


### 観測（Observation）

- **形状**: デフォルトで `(7, 7, 3)` の3次元配列（`obs_range` を変更すると `(obs_range, obs_range, 3)`）  
- **値の範囲**: 各要素は `[0, 30]` の整数  
- **チャンネルの意味**:
  - **チャンネル1（[:, :, 0]）**: 壁の有無  
    - `1` が立っている位置に壁がある
  - **チャンネル2（[:, :, 1]）**: 味方（ally）の数  
    - そのセルにいる追跡者（pursuer）の数
  - **チャンネル3（[:, :, 2]）**: 敵（opponent / evader）の数  
    - そのセルにいる逃げるエージェントの数

観測は**各エージェントを中心としたローカルなグリッド**として与えられます[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

### 報酬（Reward）

追跡者（pursuer）は、以下のような報酬を受け取ります。

- **捕獲報酬（catch_reward）**:  
  - 逃げるエージェント（evader）を**完全に包囲して捕まえたとき**、その周りにいる追跡者それぞれに **+5.0** の報酬が与えられます。

- **タッチ報酬（tag_reward）**:  
  - 追跡者が逃げるエージェントに**触れた（タグした）とき**、**+0.01** の報酬が与えられます。

- **緊急度ペナルティ（urgency_reward）**:  
  - 各ステップで **-0.1** の報酬が加算されます。  
  - これは「時間をかけるほど不利」になるように設計されたペナルティで、**素早く捕まえることを促す**役割があります。

- **共有報酬（shared_reward）**:  
  - `shared_reward=True` に設定すると、上記の報酬が**全エージェントで均等に分配**されます（協調型タスクとして扱いやすくするためのオプション）[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

### 終了条件（Done / Termination / Truncation）

エピソードは以下のいずれかの条件で終了します。

- **Termination（通常終了）**:  
  - すべての逃げるエージェント（evader）が捕まえられたとき

- **Truncation（打ち切り）**:  
  - `max_cycles`（デフォルト 500）のステップ数に達したとき

PettingZooのAPIでは、`terminated` と `truncated` を分けて返す形になっていますが、Pursuit環境では上記2パターンで終了します[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。


### Info

- `reset()` や `step()` で **info 辞書**が返されますが、Pursuit固有の特別なキーについては公式ドキュメント上で詳細に列挙されていません。  
- 一般的には、PettingZoo標準のinfo構造（環境のメタ情報など）が含まれます。

要約すると、Pursuit環境からは

- **ローカルなグリッド観測**（壁・味方・敵の位置情報）
- **捕獲・タッチ・時間ペナルティを含む報酬**
- **全evader捕獲 or 最大ステップ到達による終了フラグ**
- **標準的なinfo辞書**

が得られる、という設計になっています[PettingZoo Pursuit Docs](https://pettingzoo.farama.org/environments/sisl/pursuit)。

## 環境の導入

必要なパッケージは以下通りです。
Google Colabでインストールしてください。

```
# エミュレータとGymnasiumのインストール
!pip install gymnasium[atari]
!pip install gymnasium[accept-rom-license]
!pip install pyvirtualdisplay > /dev/null 2>&1
!pip install AutoROM[accept-rom-license]
!AutoROM --accept-license
# 描画用のシステムパッケージ
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install pettingzoo[atari,accept-rom-license]
!pip install "pettingzoo[sisl]"
!pip install supersuit
```

### 動作

エージェントをランダムに動作させるコードを作りました。
以下を上記のパッケージをインストール後に動作させて下さい。

```python
import imageio
from pettingzoo.sisl import pursuit_v4

# 動画の保存先ファイル名
video_filename = "pursuit_random_agents.mp4"

# 環境生成（render_mode="rgb_array" で画像取得可能にする）
env = pursuit_v4.env(render_mode="rgb_array")

# フレームを格納するリスト
frames = []

# 環境リセット
env.reset()

# ランダムエージェントで1エピソード実行し、各ステップの画面を保存
for agent in env.agent_iter():
    # 現在の状態を取得
    obs, reward, terminated, truncated, info = env.last()

    # 画面を画像として取得
    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        action = None
    else:
        # ランダム行動
        action = env.action_space(agent).sample()

    env.step(action)

env.close()

# フレームを動画にエンコードして保存
with imageio.get_writer(video_filename, fps=10) as video:
    for frame in frames:
        video.append_data(frame)

print(f"動画を保存しました: {video_filename}")
```

すると以下のようにエージェントの敵味方に分かれてゲームする様子が確認されます。


## 本日まとめ

ということで前回までの反省点からチームプレーが必要となるゲームの選定を行いました。
お互い密集した方が有利や、敵を分断させるなど勝率の高い動作をMARLを通じて学習できるように次回以降、学習環境を設計していきます。

