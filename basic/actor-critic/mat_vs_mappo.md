結論から言うと、**2025年時点では、MAPPO/HAPPO系の後継手法（分散化・通信・リフレクティブ最適化など）の方が、MATのような「一つの中央Transformerで全エージェントを制御する」手法よりも、研究・実装の両面で「主流」に近い状況**です。ただし、MAT系（MAT/PMATなど）も**重要な比較対象・ベースライン**として広く参照されています。

以下、理由を整理します。

---

### 1. MAPPO/HAPPO系の後継手法の広がり
- MAPPO/HAPPO は、**CTDE（Centralized Training, Decentralized Execution）** の枠組みで、
  - 中央化された批評家（critic）＋分散化されたアクター（actor）という構造が実装しやすく、
  - 既存のPPOコードベースを流用しやすいため、**実装・運用のハードルが低い**という利点があります。
- 2024–2025年の研究では、
  - **DG-MAPPO（Distributed Graph-Attention MAPPO）** のように、**完全分散化・通信ベース**のMAPPO拡張が提案されています[OpenReview](https://openreview.net/forum?id=fotzssBy3o)。
  - **MARPO（Multi-Agent Reflective Policy Optimization）** のように、PPO系の改良（リフレクティブ機構、非対称クリッピングなど）でサンプル効率と安定性を高める手法も登場しています[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40219/44180)。
- これらは、
  - **既存のMAPPO/HAPPOの枠組みを拡張する形**で提案されており、
  - 実装・比較のしやすさから、**多くのベンチマークや実システムで採用されやすい**傾向があります。

---

### 2. MAT系（中央集権Transformer）の位置づけ
- MAT は、**MARLを系列モデリング問題として扱う**という発想で、
  - 1つのエンコーダ・デコーダTransformerで全エージェントの行動を系列生成する**中央集権型**の手法です[arXiv](https://arxiv.org/abs/2205.14953)。
- 2025年には、**PMAT（Prioritized Multi-Agent Transformer）** のように、
  - MATのエンコーダ・デコーダ構造に**行動生成順序の最適化**を組み込んだ拡張も提案されています[AAMAS 2025](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p997.pdf)。
- しかし、
  - **モデル構造がやや複雑**で、Transformerの大規模化・系列長の増大に伴う計算コストが課題になりやすい。
  - 実システムでは、**分散実行（各エージェントが独立に推論）**が求められる場面が多く、中央集権型モデルをそのまま適用しづらい。
  といった理由から、**「最先端の比較対象・ベースライン」としてはよく使われるものの、「デフォルトの実装」という位置づけにはまだ至っていない**印象です。

---

### 3. サーベイ・ベンチマークでの扱い
- 2025年のMARLサーベイやベンチマーク論文では、
  - **MAPPO/HAPPO系の手法が「標準的なベースライン」として広く採用**されており、
  - MAT/PMAT は、**系列モデリングベースの新しいアプローチとして比較対象に含まれる**ことが多いです[AAMAS 2025 Benchmark](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p1613.pdf)。
- これは、
  - MAPPO/HAPPO系が**実装の安定性・再現性が高く、比較の基準として使いやすい**一方、
  - MAT系は**Transformerの表現力を活かした高性能な代替案**として評価されている、という構図を反映しています。

---

### 4. 実用上の使い分け
- **MAPPO/HAPPO系（およびその後継）**：
  - 既存のPPOコードベースを流用しやすい。
  - CTDE構造が多くの実システム（ロボット群、ゲームAIなど）と相性が良い。
  - 分散化・通信・リフレクティブ最適化など、**実用的な拡張が進みやすい**。
- **MAT/PMAT系**：
  - Transformerの表現力を活かし、**長期的依存関係や複雑な協調**を捉えやすい。
  - ただし、モデル規模・計算コスト・分散実行の難しさから、**研究段階での検証が中心**になりがち。

---

### 5. まとめ
- **「主流」という観点では、2025年時点ではMAPPO/HAPPO系の後継手法（DG-MAPPO, MARPOなど）の方が、研究・実装の両面で広く使われている**傾向があります。
- MAT/PMATのような**中央集権Transformer系の手法**は、
  - **高性能な比較対象・ベースライン**として重要な位置を占めており、
  - 「系列モデリングとしてMARLを扱う」という新しい方向性の**代表格**として認知されていますが、
  - 現時点では**デフォルトの実装としての普及度ではMAPPO/HAPPO系にやや劣る**、というのが現状です[arXiv](https://arxiv.org/abs/2205.14953)[AAMAS 2025 PMAT](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p997.pdf)[OpenReview DG-MAPPO](https://openreview.net/forum?id=fotzssBy3o)。