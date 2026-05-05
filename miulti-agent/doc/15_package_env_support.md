Google Colab で動かせる MARL 向け環境・パッケージをいくつか挙げます。いずれも公式に Colab ノートブックやチュートリアルが用意されているものです。

---

### 1. Melting Pot（DeepMind）

- **概要**: 50以上のマルチエージェント基盤（ゲーム）と 256 以上のテストシナリオからなる MARL 評価スイート。協力・競争・社会的相互作用など多様な状況をカバーします。
- **Colab 対応**: PyPI パッケージ `dm-meltingpot` として提供され、公式リポジトリに評価用ノートブックと「Open in Colab」バッジがあります。
- **主な用途**: マルチエージェントの一般化性能評価、社会的ジレンマの研究など。
- **Colab ノートブック例**:
  - `notebooks/evaluation_results.ipynb`
    → https://colab.research.google.com/github/deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb
    [Melting Pot GitHub](https://github.com/google-deepmind/meltingpot)

---

### 2. AI Economist / Foundation（Salesforce）

- **概要**: 経済シミュレーションのためのマルチエージェント環境。労働者（エージェント）と政府（社会的プランナー）の相互作用を Gym 互換 API でモデル化します。
- **Colab 対応**: `tutorials/` 以下に複数の Colab ノートブックがあり、各チュートリアルから直接 Colab で開けるリンクが用意されています。
- **主な用途**: 経済政策シミュレーション、最適課税、マルチエージェント RL の応用など。
- **Colab ノートブック例**:
  - `economic_simulation_basic.ipynb`
  - `multi_agent_training_with_rllib.ipynb`
  - `multi_agent_gpu_training_with_warp_drive.ipynb`
    [AI Economist GitHub](https://github.com/salesforce/ai-economist)

---

### 3. PettingZoo（Farama Foundation）

- **概要**: Gymnasium 風の API を持つマルチエージェント環境標準ライブラリ。AEC（順次行動）と Parallel（同時行動）の 2 形式をサポートし、Atari や古典ゲームなど多数の環境を提供します。
- **Colab 対応**: 公式ドキュメント自体は Colab を明示していませんが、`pip install pettingzoo` でインストールでき、Colab 上でも問題なく利用できます。Stable-Baselines3 や RLlib との連携チュートリアルも多く、それらを Colab に移植しやすい構成です。
- **主な用途**: 一般的な MARL 研究、カスタム環境作成、既存 RL ライブラリとの統合。
- **公式サイト**: https://pettingzoo.farama.org/
  [PettingZoo Documentation](https://pettingzoo.farama.org/index.html)

---

### 4. Mava（InstaDeep）

- **概要**: JAX ベースの分散マルチエージェント RL フレームワーク。PPO、Q-learning、SAC などのアルゴリズムを単一ファイルで実装し、研究プロトタイピングを高速化します。
- **Colab 対応**: 公式リポジトリに Quickstart 用の Colab ノートブックが用意されており、Python 3.10（Colab デフォルト）向けに調整されています。
- **主な用途**: JAX ベースの高速 MARL 実験、分散学習、新しいアルゴリズムの実装。
- **Colab ノートブック例**:
  - `examples/Quickstart.ipynb`
    → https://colab.research.google.com/github/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb
    [Mava GitHub](https://github.com/instadeepai/mava)

---

### 5. VMAS（Vectorized Multi-Agent Simulator）

- **概要**: PyTorch ベースの 2D 物理エンジン兼マルチエージェントシミュレータ。複数ロボットのナビゲーションや衝突回避など、物理的な MARL 環境を効率的にシミュレートできます。
- **Colab 対応**: 公式ドキュメントに「Notebooks」セクションがあり、VMAS 環境の作成・描画や BenchMARL（TorchRL の MARL ライブラリ）での学習など、複数の Colab ノートブックが公開されています。
- **主な用途**: 物理ベースのマルチエージェントタスク（ナビゲーション、追跡、衝突回避など）。
- **Colab ノートブック例**:
  - VMAS の基本的な使い方（環境作成・描画）
  - VMAS を BenchMARL で学習するノートブック
  - TorchRL 公式チュートリアル（`multiagent_ppo.ipynb` など）で VMAS 環境を利用
    [VMAS Documentation](https://vmas.readthedocs.io/en/stable/usage/notebooks.html)

---

### まとめ

- **評価・ベンチマーク重視**: Melting Pot、VMAS
- **経済・社会シミュレーション**: AI Economist / Foundation
- **汎用 MARL 環境と標準 API**: PettingZoo
- **JAX ベースの高速 MARL 実験**: Mava

いずれも Colab 上で `pip install` ＋公式ノートブックを開くだけで試せるようになっているので、用途に合わせて選ぶとよいと思います。
