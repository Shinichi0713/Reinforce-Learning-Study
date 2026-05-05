
最近MARLのトライアル中に自作環境で困ることが多いと感じました。(この下りは記事下の総括でも触れますが・・・結構手痛い目を見ました)
そこで、Google Colabで動作する、MARL環境を構築することが出来る、利用者が一定数以上存在するという条件でパッケージの選定を行いました。

## MARL環境調査

Google Colab で動作し、かつ利用者が一定数以上いる MARL 環境パッケージとして、代表的なものを以下に挙げ、それぞれを

- **ネット上の情報量（利用者数・認知度）**
- **パッケージの更新性（メンテナンス状況）**
- **環境の豊富さ（タスクの多様性）**

の 3 軸で評価します。

### 候補パッケージの概要と評価

__1. PettingZoo（Farama Foundation）__

- **概要**: Gymnasium 互換の API を持つマルチエージェント環境標準ライブラリ。Classic、MPE、Atari など多数の環境を提供。
- **Colab 対応**: `pip install pettingzoo["mpe"]` などでインストール可能。公式ドキュメントやチュートリアルも豊富。
- **GitHub スター数**: 10,000 以上と非常に多く、MARL 環境のデファクトスタンダードとして認知されています[PettingZoo Documentation](https://pettingzoo.farama.org/index.html)。

**評価**

- **情報量・利用者数**: ★★★★★  
  GitHub スター数が 1 万超えで、論文・ブログ・チュートリアルも非常に多い。MARL 環境の「標準 API」として扱われることが多く、情報量はトップクラス。
- **更新性**: ★★★★☆  
  Farama Foundation（旧 OpenAI Gym メンツ）がメンテナンスしており、Gymnasium との整合性も高く、定期的にアップデートされている。
- **環境の豊富さ**: ★★★★★  
  Classic（じゃんけん、囲碁、四目並べ）、MPE（粒子環境）、Atari（Pong, Space Invaders など）、さらに SMAC, SUMO, GRF などのサードパーティ環境も PettingZoo API でラップされている[PettingZoo Third-Party Environments](https://pettingzoo.farama.org/environments/third_party_envs/)。  
  協力・競争・混合インセンティブなど、タスクの幅が非常に広い。

__2. Melting Pot（DeepMind）__

- **概要**: 50 以上のマルチエージェント基盤と 256 以上のテストシナリオからなる MARL 評価スイート。社会的相互作用（協力・競争・欺瞞など）の一般化性能を測るベンチマーク[Melting Pot GitHub](https://github.com/google-deepmind/meltingpot)。
- **Colab 対応**: `dm-meltingpot` として PyPI に公開され、評価用ノートブックが Colab で開けるようになっている。
- **GitHub スター数**: 約 750 前後と、研究用途に特化した中規模プロジェクト[Top 10 MARL Repositories](https://medium.com/@gwrx2005/top-10-github-repositories-for-multi-agent-reinforcement-learning-marl-platforms-05cc8d21a6c1)。

**評価**

- **情報量・利用者数**: ★★★☆☆  
  研究コミュニティでは有名だが、一般的な MARL 入門用途での利用は少なめ。GitHub スターも PettingZoo の 1/10 程度。
- **更新性**: ★★★☆☆  
  DeepMind が開発・公開しているが、PettingZoo ほど頻繁な更新はなく、研究ベンチマークとしての位置づけが強い。
- **環境の豊富さ**: ★★★★☆  
  社会的ジレンマに特化した多様なシナリオ（協力・競争・欺瞞・信頼など）が揃っているが、タスクの種類は「社会的相互作用」に集中している。

__3. AI Economist / Foundation（Salesforce）__

- **概要**: 経済シミュレーションのためのマルチエージェント環境。労働者と政府（社会的プランナー）の相互作用を Gym 互換 API でモデル化[AI Economist GitHub](https://github.com/salesforce/ai-economist)。
- **Colab 対応**: `tutorials/` に複数の Colab ノートブックがあり、GPU トレーニング（WarpDrive）や RLlib 連携などが紹介されている。
- **GitHub スター数**: 数百〜千程度で、経済・政策シミュレーションに特化したニッチな利用。

**評価**

- **情報量・利用者数**: ★★☆☆☆  
  経済政策・エネルギー市場などの応用研究では使われるが、一般的な MARL 入門では利用が限定的。
- **更新性**: ★★☆☆☆  
  プロジェクトは公開されているが、PettingZoo ほど活発なコミュニティ更新は見られない。アーカイブ的な扱いになりつつある。
- **環境の豊富さ**: ★★☆☆☆  
  経済シミュレーションに特化しており、タスクの種類は限定的。汎用 MARL 環境としては狭い。

__4. VMAS（Vectorized Multi-Agent Simulator）__

- **概要**: PyTorch ベースの 2D 物理エンジン兼マルチエージェントシミュレータ。ベクトル化された環境で効率的な MARL ベンチマークを提供[VMAS GitHub](https://github.com/proroklab/vectorizedmultiagentsimulator)。
- **Colab 対応**: BenchMARL（TorchRL の MARL ライブラリ）と連携した Colab ノートブックが公式に提供されている。
- **GitHub スター数**: 数百程度で、研究用途の高性能シミュレータとして認知されている。

**評価**

- **情報量・利用者数**: ★★☆☆☆  
  高性能シミュレータとして論文でよく引用されるが、一般的な MARL 入門での利用は少ない。
- **更新性**: ★★★☆☆  
  研究プロジェクトとして継続的に開発されているが、PettingZoo ほどの大規模コミュニティではない。
- **環境の豊富さ**: ★★★☆☆  
  物理ベースのマルチロボットタスク（ナビゲーション、追跡、衝突回避など）に強みがあるが、ゲームや経済など他ドメインはカバーしていない。

__5. Mava（InstaDeep）__

- **概要**: JAX ベースの分散 MARL フレームワーク。PPO, Q-learning, SAC などのアルゴリズムを単一ファイル実装で提供し、高速な研究プロトタイピングを支援[Mava GitHub](https://github.com/instadeepai/mava)。
- **Colab 対応**: Quickstart 用の Colab ノートブックが公式に用意されている。
- **GitHub スター数**: 数百程度で、JAX ベース MARL の代表的なフレームワーク。

**評価**

- **情報量・利用者数**: ★★☆☆☆  
  JAX コミュニティでは有名だが、一般的な MARL 入門では PettingZoo ほど広くは使われていない。
- **更新性**: ★★★☆☆  
  InstaDeep がメンテナンスしており、研究用途としては活発。
- **環境の豊富さ**: ★★☆☆☆  
  主にアルゴリズム実装にフォーカスしており、環境自体は PettingZoo や Melting Pot をラップして使うことが多い。独自環境の数は限定的。

### 総合評価と PettingZoo の選定理由
ということでパッケージの評価結果を以下に示します。

| パッケージ       | 情報量・利用者数 | 更新性 | 環境の豊富さ | 合計（★15点満点） |
|------------------|------------------|--------|--------------|-------------------|
| PettingZoo       | 5                | 4      | 5            | **14**            |
| Melting Pot      | 3                | 3      | 4            | 10                |
| AI Economist     | 2                | 2      | 2            | 6                 |
| VMAS            | 2                | 3      | 3            | 8                 |
| Mava            | 2                | 3      | 2            | 7                 |

※あくまで相対的な目安です。


### 結論

- **Melting Pot** は社会的ジレンマの評価に強みがあり、**VMAS** は物理ベースの高性能シミュレーションに優れ、**Mava** は JAX ベースの高速アルゴリズム実装に特化しています。  
- しかし、**Google Colab で MARL 環境を構築し、利用者が一定数以上いる**という条件を満たし、かつ情報量・更新性・環境の豊富さのバランスが最も良いのは **PettingZoo** です。

したがって、**「Colab で動かせる MARL 環境パッケージを 1 つ選ぶなら PettingZoo が最適」** と考えました。

## PettingZooとは

PettingZoo は、**マルチエージェント強化学習（MARL）用の環境を集めた Python パッケージ**です。  
Farama Foundation（OpenAI Gym や Gymnasium を開発している団体）が開発・メンテナンスしています。

### 1. PettingZoo は何をするパッケージか

- **目的**:  
  単一エージェント向けの Gym/Gymnasium のように、**マルチエージェント強化学習の環境を標準化し、共通の API で扱えるようにする**ことです。
- **提供物**:
  - 多数のマルチエージェント環境（古典ゲーム、Atari、MPE など）
  - それらを扱うための統一的な Python インターフェース
  - カスタム環境を作るためのテンプレートやユーティリティ


### 2. 主な特徴

__(1) Gymnasium 互換の API 設計__

- **AEC（Agent Environment Cycle）API**  
  エージェントが順番に行動する「ターン制」のゲーム向けです。  
  - `env.reset()` → `for agent in env.agent_iter():` → `env.last()` → `env.step(action)` という流れで制御します。
- **Parallel API**  
  全エージェントが同時に行動する「同時行動」型のゲーム向けです。  
  - `observations = env.reset()` → `actions = {agent: ...}` → `observations, rewards, ... = env.step(actions)` という形式です。

どちらの API も Gym/Gymnasium の `reset` / `step` に似た感覚で使えるため、既存の RL ライブラリとの統合がしやすいです。

__(2) 豊富な環境コレクション__

代表的なカテゴリ：

- **Classic（古典ゲーム）**  
  じゃんけん（`rps_v2`）、四目並べ（`connect_four_v3`）、囲碁（`go_v5`）など。
- **MPE（Multi-Agent Particle Environments）**  
  スピーカー・リスナー、追跡・回避、協力ナビゲーションなど、2D の物理ベース環境。
- **Atari（マルチエージェント版）**  
  Pong、Space Invaders などを複数エージェントでプレイする環境。
- **その他**  
  カードゲーム、ボードゲーム、自作環境なども統合可能。

__(3) マルチエージェントに特化した設計__

- 各エージェントごとに `observation_space`、`action_space`、`reward` を持てる。
- エージェントの参加・離脱（エピソード中にエージェント数が変わる）も扱える。
- 部分観測（Partial Observability）や非対称な役割など、MARL 特有の設定を自然に表現できます。

__(4) 主要 RL ライブラリとの連携__

PettingZoo は「環境だけ」を提供し、学習アルゴリズムは別ライブラリに任せる設計です。  
公式ドキュメントには以下の連携例があります：

- Stable-Baselines3
- Ray RLlib
- Tianshou
- CleanRL
- AgileRL

これらと組み合わせることで、PettingZoo 環境上で PPO、DQN などのアルゴリズムをすぐに試せます。

__(5) カスタム環境の作成が容易__

- Gymnasium と同様に、`Env` クラスを継承して独自の MARL 環境を実装できます。
- AEC / Parallel のどちらか一方を実装すれば、もう一方は自動変換するユーティリティも用意されています。


## 動作させる

PettingZoo で MARL 環境をランダムに動かす代表的なコードを、AEC 形式と Parallel 形式の両方で示します。

因みに、動作で困ったら[公式リファレンス](https://pettingzoo.farama.org/content/environment_creation/)が一番頼りになります。
強化学習のパッケージは知らない間に改版され、従来コードが動かないことが多々あり、ネットや生成AIがあんまり役に立たないことがあります。

今回は、Google Colabはレンダリングをしてくれない、でも、動作確認はしたいと考えました。
そこで**PettingZoo の Atari 環境（例: `pong_v3`）を動作して、結果を OpenCV で動画保存する例**を示します。

### OpenCV でフレームをキャプチャして動画保存する例

__準備：必要なライブラリのインストール__

```python
!pip install pettingzoo["atari"] opencv-python
!AutoROM --accept-license  # Atari ROM のインストール
```

__動画保存コード（Pong 環境を例に）__

```python
import cv2
import numpy as np
from pettingzoo.atari import pong_v2

# 環境生成（human モードで描画）
env = pong_v2.env(render_mode="rgb_array")  # rgb_array でフレーム取得
env.reset()

# 動画保存用の設定
fps = 30
frame_size = (env.unwrapped.screen_width, env.unwrapped.screen_height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("pong_episode.mp4", fourcc, fps, frame_size)

# エピソードを実行し、各ステップの画面を動画に書き込む
max_steps = 500
step = 0

for agent in env.agent_iter():
    if step >= max_steps:
        break

    obs, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

    # 現在の画面を RGB 配列として取得
    frame = env.render()
    # OpenCV は BGR 形式なので変換
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

    step += 1

env.close()
video_writer.release()

print("動画を保存しました: pong_episode.mp4")
```

__動画のダウンロード__

Colab 左側のファイルブラウザ（📁）から `pong_episode.mp4` を右クリック →「ダウンロード」でローカルに保存できます。

こんな感じの動作が得られました。

<img src="images/2_intro_petting_zoo/pong_episode.mp4" width="600">

## 総括
MARLの環境を構築するのがいつも手間だった、そして、うまく解けない場合、自分で作った環境の場合、情報がなかなかないということから、パッケージを使ったMARL環境の構築を行いました。

出来れば今後は今回導入したMARL環境でトライアルをしていきたいと思います。

