ここでは、**不偏分散（unbiased sample variance）** の公式

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
$$

が、なぜ $n$ ではなく $n-1$ で割るのかを、期待値の計算から簡潔に導出します。



### 1. 前提

- 母集団の平均：$\mu$
- 母分散：$\sigma^2 = E[(X-\mu)^2]$
- 独立同分布な標本：$x_1, x_2, \dots, x_n$
- 標本平均：$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

目標：  
$$
E\left[ \sum_{i=1}^n (x_i - \bar{x})^2 \right] = (n-1)\sigma^2
$$
を示し、その結果から不偏分散の公式を導く。



### 2. 偏差平方和の分解

まず、各 $x_i$ と母平均 $\mu$ の差を分解します。

$$
x_i - \bar{x} = (x_i - \mu) - (\bar{x} - \mu)
$$

これを2乗して和を取ると：

$$
\sum_{i=1}^n (x_i - \bar{x})^2
= \sum_{i=1}^n \left[ (x_i - \mu) - (\bar{x} - \mu) \right]^2
$$

展開すると：

$$
= \sum_{i=1}^n (x_i - \mu)^2
- 2(\bar{x} - \mu) \sum_{i=1}^n (x_i - \mu)
+ \sum_{i=1}^n (\bar{x} - \mu)^2
$$

ここで、

- $\sum_{i=1}^n (x_i - \mu) = n(\bar{x} - \mu)$
- $\sum_{i=1}^n (\bar{x} - \mu)^2 = n(\bar{x} - \mu)^2$

なので、

$$
\sum_{i=1}^n (x_i - \bar{x})^2
= \sum_{i=1}^n (x_i - \mu)^2 - n(\bar{x} - \mu)^2
$$



### 3. 期待値を取る

両辺の期待値を取ります。

$$
E\left[ \sum_{i=1}^n (x_i - \bar{x})^2 \right]
= E\left[ \sum_{i=1}^n (x_i - \mu)^2 \right]
- n E\left[ (\bar{x} - \mu)^2 \right]
$$

- 各 $x_i$ は独立で、$E[(x_i - \mu)^2] = \sigma^2$ なので、
  $$
  E\left[ \sum_{i=1}^n (x_i - \mu)^2 \right] = n\sigma^2
  $$

- 標本平均の分散は $ \mathrm{Var}(\bar{x}) = \frac{\sigma^2}{n} $ なので、
  $$
  E[(\bar{x} - \mu)^2] = \mathrm{Var}(\bar{x}) = \frac{\sigma^2}{n}
  $$

したがって、

$$
E\left[ \sum_{i=1}^n (x_i - \bar{x})^2 \right]
= n\sigma^2 - n \cdot \frac{\sigma^2}{n}
= (n-1)\sigma^2
$$



### 4. 不偏分散の公式の導出

上の結果から、

$$
E\left[ \sum_{i=1}^n (x_i - \bar{x})^2 \right] = (n-1)\sigma^2
$$

です。  
したがって、

$$
E\left[ \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2 \right] = \sigma^2
$$

となります。  
つまり、

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
$$

は、**母分散 $\sigma^2$ の不偏推定量**です。



### 5. まとめ

- 標本平均 $\bar{x}$ を使うと、偏差平方和 $\sum (x_i - \bar{x})^2$ の期待値は $(n-1)\sigma^2$ になる。
- よって、$n-1$ で割った量 $s^2$ の期待値は $\sigma^2$ に一致する。
- これが、**不偏分散の公式** $ s^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2 $ の導出です。

この導出は、「標本平均はデータに合わせて動くため、偏差平方和が少し小さめに出る」という性質を、期待値の計算で補正した結果と解釈できます。