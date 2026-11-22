
ここまでマルチエージェントの実装はほぼ[MAPPO](https://yoshishinnze.hatenablog.com/entry/2026/01/10/153824)を用いてきましたが、そろそろ別のベースラインとなるアルゴリズムを考えたいと思います。

ということでMAPPOの正当な後継手法を調べているとより良いと感じた方法が見つかりました。
本日はそんな手法であるA2POという手法について説明していきます。

## 概要

本日テーマに挙げるA2PO（Agent-by-Agent Policy Optimization）の論文明細は以下の通りです。

### A2PO 論文明細

| 項目 | 内容 |
|------|------|
| **論文タイトル** | Order Matters: Agent-by-agent Policy Optimization |
| **著者** | Xihuai Wang, Zheng Tian, Ziyu Wan, Ying Wen, Jun Wang, Weinan Zhang |
| **著者所属** | Shanghai Jiao Tong University, Digital Brain Lab, ShanghaiTech University, University College London |
| **発表年** | 2023年 |
| **採択学会** | **ICLR 2023**（International Conference on Learning Representations） |
| **発表形式** | Poster |
| **arXiv** | arXiv:2302.06205 |

[ICLR 2023 OpenReview](https://openreview.net/forum?id=Q-neeWNVv1)
[A2PO arXiv論文](https://arxiv.org/abs/2302.06205)
[A2PO 公式実装リポジトリ](https://github.com/xihuai18/A2PO-ICLR2023)
[ICLR 2023 Virtual Poster](https://iclr.cc/virtual/2023/poster/)

### 補足

本論文では、MAPPO・CoPPO・HAPPO・A2POの4つのCTDE（Centralized Training with Decentralized Execution）マルチエージェント強化学習アルゴリズムを統一的な枠組みで実装・比較しており、A2POが主たる研究貢献となっています。特に、エージェントごとの逐次更新における単調改善保証と、更新順序の最適化（semi-greedy selection rule）が主要な理論的貢献です。

## 本手法が課題とした内容

A2POが解決しようとした課題は、主に以下の3つです。

### 1. 逐次更新スキームにおけるサンプル非効率性

従来の逐次更新手法（例: RPISA）では、エージェントごとに別々のロールアウトが必要で、n体のエージェントに対してn回のロールアウトが必要となり、サンプル効率が極めて悪いという問題がありました。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「RPISA effectively turns non-stationary MARL problems into stationary single-agent RL ones. However, it is sample-inefficient since it only utilizes 1/n of the collected samples to update n agents' policies.」

A2POは、**1回のロールアウトサンプルから全エージェントのポリシーを逐次更新**できるようにすることで、この問題を解決しました。

### 2. 逐次更新における各エージェントの単調改善保証の欠如

HAPPOは共同ポリシーの単調改善は保証されますが、**各エージェント個別のポリシー改善が理論的に保証されていない**という問題がありました。これにより、あるエージェントの更新が先行エージェントの改善を相殺してしまう可能性がありました。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「HAPPO does not consider the distribution shift caused by preceding agents, and has no monotonic policy improvement guarantee for each agent's policy.」

A2POは、**PreOPC（preceding-agent off-policy correction）** という新しいオフポリシー補正手法を提案し、**共同ポリシーと各エージェントのポリシーの両方に対する単調改善保証**を初めて実現しました。

### 3. エージェントの更新順序の影響

逐次更新スキームでは、**どの順序でエージェントを更新するかが性能に大きく影響**するという課題がありましたが、従来の手法ではこの順序を適切に決定する枠組みがありませんでした。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「From the perspective of sequentially updating agents, we further consider the effect of agent updating order and extend the theory of non-stationarity into the sequential update scheme.」

A2POは、**semi-greedyなエージェント選択ルール**と**適応的クリッピングパラメータ**を導入し、更新順序を最適化することで非定常性問題を緩和しました。

課題と解決法を一覧としてまとめるとこんな感じです。

| 課題 | 従来手法の問題 | A2POの解決策 |
|------|-------------|-------------|
| サンプル非効率性 | RPISAはn回のロールアウトが必要 | 1回のロールアウトで全エージェントを逐次更新 |
| 個別エージェントの改善保証なし | HAPPOは共同ポリシーのみ保証 | PreOPCによる各エージェントの単調改善保証 |
| 更新順序の影響 | 順序を適切に決定する枠組みなし | semi-greedy選択ルールと適応的クリッピング |

## 提案手法

A2PO（Agent-by-Agent Policy Optimization）で提案されている手法は、主に以下の4つの要素から構成されます。それぞれの理論的根拠と合わせて説明します。

### 1. PreOPC（Preceding-agent Off-Policy Correction）

__提案手法__

各エージェント $i$ を更新する際に、先行エージェント $1, \dots, i-1$ のポリシー変更によって生じる分布シフトを補正するためのオフポリシー補正項を導入します。

__理論的根拠__

逐次更新では、エージェント $i$ を更新する時点で、先行エージェントのポリシーは既に新しいもの（$\pi'_{1:i-1}$）に更新されていますが、収集したサンプルは古い共同ポリシー（$\pi_{1:i-1}$）から得られています。このミスマッチを補正しないと、重要性サンプリング比率が偏ってしまい、各エージェントのポリシー改善が理論的に保証できません。

A2POは、エージェント $i$ の目的関数を以下のように定式化します：

$$\mathcal{L}^{A2PO}_i(\theta_i) = \mathbb{E}_{(s,a) \sim \pi} \left[ \frac{\pi'_{1:i-1}(a_{1:i-1}|s)}{\pi_{1:i-1}(a_{1:i-1}|s)} \cdot \frac{\pi_{\theta_i}(a_i|s)}{\pi_i(a_i|s)} \cdot A_{\pi}(s,a) \right]$$

ここで、$\frac{\pi'_{1:i-1}(a_{1:i-1}|s)}{\pi_{1:i-1}(a_{1:i-1}|s)}$ が**先行エージェントのポリシー変更による補正項**です。

__理論的成果__

この補正により、**共同ポリシーだけでなく、各エージェント個別のポリシーに対しても単調改善保証**が成立することが証明されました。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「We propose a preceding-agent off-policy correction (PreOPC) to guarantee the monotonic policy improvement of each agent's policy.」

### 2. Semi-Greedyなエージェント選択ルール

__提案手法__

エージェントを固定順序ではなく、各イテレーションで**期待改善量が大きいエージェントを優先的に選ぶ**greedyな選択ルールを導入します。ただし、純粋なgreedyでは探索が不十分になるため、**semi-greedy**（ε-greedy的な）アプローチを採用します。

__理論的根拠__

理論的には、**更新順序が性能に影響を与える**ことが示されています。先行エージェントほど他のエージェントに影響を与えるため、より大きな改善が期待できるエージェントを先に更新することで、共同ポリシー全体の改善を最大化できます。

__理論的成果__

論文では、**greedyな順序選択が共同ポリシーの改善を最大化する**ことを理論的に示しています。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「We propose a semi-greedy agent selection rule to optimize the agent updating order, which can maximize the joint policy improvement.」

### 3. Adaptive Clipping Parameter

__提案手法__

PPOのクリッピングパラメータ $\epsilon$ をエージェントごと・イテレーションごとに適応的に調整します。

__理論的根拠__

逐次更新では、先行エージェントの更新によって状態訪問分布が変化し、後続エージェントの更新において重要性サンプリング比率の分散が増大します。固定の $\epsilon$ では、この変化に対応できず、性能が劣化する可能性があります。

__理論的成果__

適応的な $\epsilon$ により、**各エージェントの更新時におけるKLダイバージェンスの変化を制御**し、理論的な改善保証を実用的な設定で維持します。

### 4. 単調改善保証（Monotonic Policy Improvement Guarantee）

__理論的定理__

A2POの主要な理論的成果は、以下の定理です：

**定理（A2POの単調改善保証）**: 適切な補正とクリッピングを用いることで、A2POは以下を満たす：

$$J(\pi') - J(\pi) \geq \sum_{i=1}^n \left[ \mathcal{L}^{A2PO}_i(\theta'_i) - C \cdot D_{KL}(\pi'_i \| \pi_i) \right]$$

ここで $C$ は定数、$D_{KL}$ はKLダイバージェンスです。これにより、各エージェントのポリシー更新が共同ポリシーの期待累積報酬を単調に改善することが保証されます。

さらに重要な点として、**各エージェント個別のポリシー $\pi_i$ に対しても単調改善が保証される**ことが、PreOPCの導入により初めて実現しました。

[A2PO論文（arXiv）](https://arxiv.org/html/2302.06205v2)

> 「To the best of our knowledge, A2PO is the first actor-critic algorithm in MARL that can guarantee the monotonic policy improvement of each agent's policy.」

### 提案手法一覧

| 要素 | 理論的根拠 | 解決する問題 |
|------|-----------|------------|
| PreOPC | 先行エージェントの分布シフト補正 | 各エージェントの単調改善保証 |
| Semi-greedy選択 | 更新順序が性能に影響 | 非定常性の緩和と改善最大化 |
| Adaptive clipping | KLダイバージェンスの制御 | 実用的な学習の安定性 |
| 単調改善定理 | 重要性サンプリング + 補正 | 理論的保証の提供 |

A2POは、これらの要素を統合することで、**1回のロールアウトから全エージェントを逐次更新しつつ、共同ポリシーと各エージェントの両方に対する単調改善保証を持つ初めてのMARLアクタークリティックアルゴリズム**となっています。

## 実験

A2PO論文で報告されている実験内容について、問題設定と結果をまとめます。

### 実験の問題設定

__使用ベンチマーク（4つ）__

| ベンチマーク | 環境の特徴 | 評価シナリオ例 |
|-------------|----------|--------------|
| **StarCraftII Multi-agent Challenge (SMAC)** | 離散行動空間、部分観測、協調戦闘シミュレーション | 2s3z, 3s5z, MMM2 など計9マップ |
| **Multi-agent MuJoCo (MA-MuJoCo)** | 連続行動空間、ロボット関節の協調制御 | エージェント数・関節数が異なる複数シナリオ |
| **Multi-agent Particle Environment (MPE)** | 単純な粒子エージェントの協調タスク | Cooperative Break Through, Serve, Pass and Shoot |
| **Google Research Football (GRF)** | フルゲームサッカーシミュレーション | 5-vs-5, 11-vs-11 フルゲームシナリオ |

__比較ベースライン__

- **MAPPO**: 同時更新型のマルチエージェントPPO
- **CoPPO**: 協調クリッピングを導入したPPO
- **HAPPO**: 逐次更新型だが個別エージェントの改善保証なし
- **IPPO**: Independent PPO（独立学習）

このベースラインとの比較結果は以下のように説明されています。

![1790476051127](image/marl_A2PO/1790476051127.png)

学習曲線だとこんな感じで立ち上がりから有効であることが確認されます。

![alt text](image.png)

### 実験結果の概要

__1. SMAC（StarCraftII）での結果__

A2POは9マップ中**6マップで（ほぼ）100%勝率**を達成し、**平均勝率96.9%**を記録しました。

| アルゴリズム | 平均勝率 |
|------------|---------|
| A2PO | **96.9%** |
| MAPPO | 88.7% |
| HAPPO | 81.9% |
| CoPPO | それ以下 |

特に難易度の高いマップ（MMM2など）でも、ベースラインを大きく上回る勝率を示しました。
という結果は👆の表にまとめたものです。

[Linerレビュー](https://liner.com/review/order-matters-agentbyagent-policy-optimization)

__2. MA-MuJoCoでの結果__

- **タスクの複雑性に応じた優位性の拡大**: ロボットの関節数が増え、連続制御の複雑さが高まるにつれて、A2POがベースラインに対するリードを拡大しました。
- これは、PreOPCによる補正と逐次更新スキームが、複雑な連続制御マルチエージェント環境の課題に対してより効果的であることを示しています。

__3. GRF（Google Research Football）での結果__

- **サンプル効率の優位性**: 5-vs-5および11-vs-11の両シナリオで、A2POはベースラインより**少ない環境ステップ数で高い勝率に到達**し、それを維持しました。
- フルゲームサッカーという高度に複雑な協調タスクにおいても、逐次更新の利点が発揮されました。

__4. MPEでの結果__

- **Cooperative Break Through**および**Serve, Pass and Shoot**タスクで、A2POは明示的な協調行動を学習できることを示しました。
- エージェント間の役割分担と連携が、逐次更新スキームによって促進されることが観察されました。

### 研究質問（RQ）に対する回答

| 研究質問 | 結論 |
|---------|------|
| **RQ1**: 性能とサンプル効率の比較 | 全ベンチマークでA2POがベースラインを大差で上回る。性能とサンプル効率の両方で優位。 |
| **RQ2**: 協調行動の促進 | A2POは複雑な協調タスクで、エージェント間の協調を促進する利点がある。 |
| **RQ3**: 各要素（PreOPC, semi-greedy, adaptive clipping）の寄与 | 各要素が全体の性能向上に寄与することをアブレーション実験で確認。 |

### 実験結果のまとめ

論文の実験により、以下が実証されました：

1. **一貫した優位性**: 離散・連続、小規模・大規模を問わず、全4ベンチマークでA2POが既存手法を上回る。
2. **複雑性に応じたスケーリング**: タスクが複雑になるほど、A2POの理論的優位性（よりタイトな単調改善境界）が実用的な性能差として現れる。
3. **サンプル効率**: 1回のロールアウトで全エージェントを更新する設計が、データ収集コストの面で有利に働く。
4. **学習の安定性**: 各エージェントの単調改善保証が、学習プロセスの安定性と最終性能の向上に寄与。

これらの結果は、A2POが単なる理論的な興味の対象ではなく、実用的なマルチエージェント強化学習タスクで広く有効なアルゴリズムであることを示しています。

## 総括

A2POの本質を、以下の5つの観点で総括します。

### 1. A2POとは何か（一言で）

> **「1回のデータ収集で、エージェントを順番に1体ずつ更新しながら、各エージェントの改善を理論的に保証する」** マルチエージェントPPOの改良版。

### 2. 何が新しいのか（既存手法との違い）

| 既存手法 | 問題点 | A2POの解決 |
|---------|--------|-----------|
| **MAPPO/CoPPO** | 全エージェント同時更新 → 非定常性問題 | 逐次更新で他エージェントの変化を認識 |
| **RPISA** | 逐次更新だが、エージェント数分のロールアウトが必要 → サンプル非効率 | 1回のロールアウトで全エージェント更新 |
| **HAPPO** | 逐次更新かつ1回のロールアウトだが、**各エージェント個別の改善保証なし** | **PreOPC**で各エージェントの単調改善を保証 |

### 3. なぜ強いのか（理論的根拠の核心）

A2POの強さは、**「先行エージェントのポリシー変更による分布シフトを補正する」** 点にあります。

- エージェント $i$ を更新する時点では、先行エージェント $1 \sim i-1$ は既に新しいポリシーに更新済み
- しかし、サンプルは古いポリシーで収集されたもの
- このミスマッチを **PreOPC（先行エージェント補正項）** で修正

$$\mathcal{L}^{A2PO}_i(\theta_i) = \mathbb{E}\left[ \underbrace{\frac{\pi'_{1:i-1}}{\pi_{1:i-1}}}_{\text{PreOPC補正}} \cdot \frac{\pi_{\theta_i}}{\pi_i} \cdot A_{\pi} \right]$$

この補正により、**各エージェントのポリシー改善が共同ポリシーの改善に確実に寄与する**ことが理論的に保証されます。

### 4. 実験で何が示されたか（核心のみ）

- **SMAC**: 9マップ中6マップで（ほぼ）100%勝率、平均勝率 **96.9%**（MAPPO 88.7%）
- **MA-MuJoCo**: タスクが複雑になるほどA2POの優位性が拡大
- **GRF**: 少ないサンプル数で高い勝率に到達（サンプル効率の優位性）
- **MPE**: 明示的な協調行動（役割分担・連携）が学習されることを確認

### 5. 実用上の意味

A2POを使うべき場面：

- **協調が重要なタスク**（戦闘、サッカー、ロボット制御など）
- **サンプル収集が高価な環境**（シミュレータが重い、実機実験など）
- **学習の安定性が重要な場面**（理論的保証に基づく安定した改善）

### 一言でまとめると

> **「逐次更新の利点（非定常性の緩和）を活かしつつ、1回のロールアウトで全エージェントを更新するサンプル効率を維持し、かつ各エージェントの改善を理論的に保証する」** という、これまでのトレードオフを解消したアルゴリズム。

MAPPOからの移行を検討する上で、**理論的保証と実用的な性能の両方を得られる正当な後継手法**と言えます。
