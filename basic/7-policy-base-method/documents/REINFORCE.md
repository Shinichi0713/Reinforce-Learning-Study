## ロス関数
REINCORCEのロス関数は強化学習においてエージェントがとった行動の確率を報酬に応じて増減する、ということを目的として設計されている。
報酬の高かった行動は今後もとりやすくし、報酬が低かった行動はとりにくくする。

## 数式的な背景

REINFORCEアルゴリズムは、方策勾配法（Policy Gradient Method）の一種です。  
エージェントの方策（policy）\(\pi_\theta(a|s)\)のパラメータ\(\theta\)を、**期待される報酬**を最大化する方向に更新します。

### 期待報酬の最大化
\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
\]
ここで\(\tau\)はエピソード（状態・行動の列）、\(R(\tau)\)は累積報酬。

### 方策勾配定理
\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]
\]
この勾配を**確率的勾配降下法**で最適化します。


