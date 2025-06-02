以下の通り、Markdownの数式表記（LaTeX記法）でまとめます。

---

### 普通のQ学習における価値関数

普通のQ学習における価値関数 $V$ は、次のように表されます。

$$
V(s_t) = \mathbb{E}_\pi \left[ Q(s_t, a_t) \right]
$$

---

### Soft-Q学習における価値関数

Soft-Q学習における価値関数 $V_{\mathrm{soft}}$ は、エントロピー項を含めて次のように表されます。

$$
V_{\mathrm{soft}}(s_t) = \mathbb{E}_\pi \left[ -\alpha \log \pi(a_t \mid s_t) + Q(s_t, a_t) \right]
$$

このように、状態価値 $V$ にエントロピーという価値が追加されています。

---

### Soft-Q学習のベルマン方程式

この式は  
「Soft-Q学習における即時報酬 ＝ 通常の即時報酬 ＋ 遷移先でのエントロピーボーナス」  
と捉えると、普通のQ学習のベルマン方程式と同じ形になります。

$$
Q_{\mathrm{soft}}(s_t, a_t) = R_{\mathrm{soft}}(s_t, a_t) + \mathbb{E}_\pi \left[ Q_{\mathrm{soft}}(s_{t+1}, a_{t+1}) \right]
$$

ここで、

$$
R_{\mathrm{soft}}(s_t, a_t) = R(s_t, a_t) + \mathbb{E}_\pi \left[ -\alpha \log \pi(a_{t+1} \mid s_{t+1}) \right]
$$

---

このように、Soft-Q学習では遷移先でのエントロピーボーナスが即時報酬に加算されている点が、通常のQ学習との違いです。