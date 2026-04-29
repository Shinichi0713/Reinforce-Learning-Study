Google Colab でデバッグする主な方法は、大きく分けて次の3つです。

1. **Python標準の `print` と `logging` を使う方法**  
   最もシンプルで、どんな環境でも使えます。

2. **Colabの「デバッガ」機能を使う方法**  
   セル単位でブレークポイントを設定してステップ実行できます。

3. **外部ツールやライブラリを使う方法**  
   `pdb` や `ipdb` など、対話的にデバッグする方法です。

---

## 1. `print` と `logging` によるデバッグ

### `print` を使う
```python
def some_function(x):
    print("x =", x)  # ここで変数の値を確認
    result = x * 2
    print("result =", result)
    return result

some_function(5)
```

- メリット：簡単で直感的
- デメリット：出力が増えると見づらくなる

### `logging` を使う
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def some_function(x):
    logger.info("x = %s", x)
    result = x * 2
    logger.info("result = %s", result)
    return result

some_function(5)
```

- ログレベル（`DEBUG`, `INFO`, `WARNING`, `ERROR`）を切り替えられるので、本番環境でも使いやすいです。

---

## 2. Colabの「デバッガ」機能を使う

Colabにはセル単位でブレークポイントを設定できるデバッガが用意されています。

### 使い方の流れ

1. **デバッグモードをオンにする**  
   メニューバーから  
   **実行 → デバッグモードをオンにする**  
   を選択します。

2. **ブレークポイントを設定する**  
   セルの左側（行番号の左あたり）をクリックすると、赤い丸（ブレークポイント）が表示されます。  
   そのセルを実行すると、その行で一時停止します。

3. **ステップ実行する**  
   ブレークポイントで止まったら、右側に表示されるデバッグパネルで  
   - 「ステップオーバー」（次の行へ）
   - 「ステップイン」（関数の中へ）
   - 「ステップアウト」（関数から抜ける）
   などのボタンで1行ずつ実行できます。

4. **変数の値を確認する**  
   デバッグパネルの「変数」タブで、現在のスコープにある変数とその値が確認できます。

### 例
```python
def add(a, b):
    return a + b

x = 10
y = 20
# ここにブレークポイントを設定
z = add(x, y)
print(z)
```

- ブレークポイントを `z = add(x, y)` の行に置いて実行すると、その行で止まり、`x`, `y` の値や `add` の戻り値を確認できます。

---

## 3. `pdb` / `ipdb` を使った対話的デバッグ

Python標準の `pdb` や、より使いやすい `ipdb` を使うと、対話的にデバッグできます。

### `pdb` の基本的な使い方

```python
import pdb

def buggy_function(x):
    pdb.set_trace()  # ここで実行が止まる
    result = x + "10"  # 型エラーが出そうなコード
    return result

buggy_function(5)
```

- `pdb.set_trace()` を実行すると、その行で一時停止し、コンソールで対話的にデバッグできます。
- 主なコマンド：
  - `n`（next）：次の行へ
  - `s`（step）：関数の中へ入る
  - `c`（continue）：次のブレークポイントまで進む
  - `p <変数名>`：変数の値を表示
  - `q`：デバッグを終了

### `ipdb` を使う（より便利）

```python
!pip install ipdb
```

```python
import ipdb

def buggy_function(x):
    ipdb.set_trace()
    result = x + "10"
    return result

buggy_function(5)
```

- `ipdb` はタブ補完やシンタックスハイライトなど、よりリッチな対話環境を提供します。

---

## 4. エラーメッセージの読み方

Colabでエラーが出たときは、**トレースバック（Traceback）** をよく見るのが重要です。

例：
```python
def divide(a, b):
    return a / b

divide(10, 0)
```

実行すると：
```
ZeroDivisionError: division by zero
```

- エラーの種類（`ZeroDivisionError`）と、どの行で発生したかが表示されます。
- トレースバックを上から順に読むと、どの関数から呼ばれて、どこで失敗したかが分かります。

---

## 5. よくあるデバッグのコツ

- **小さく分けて確認する**  
  大きな関数を一度に動かすのではなく、小さな単位で動作を確認します。

- **変数の値をこまめに確認する**  
  `print` やデバッガで、期待通りの値になっているか確認します。

- **エラーメッセージをコピペして検索する**  
  よく分からないエラーが出たら、エラーメッセージをそのままWeb検索すると、同じ問題に遭遇した人の解決策が見つかることが多いです。

---

## まとめ

- まずは `print` で簡単に確認し、必要に応じてデバッガや `pdb`/`ipdb` を使うのがおすすめです。
- Colabのデバッガはセル単位でブレークポイントを設定でき、ステップ実行や変数確認ができるので、複雑なコードのデバッグに便利です。
- エラーメッセージを丁寧に読むことで、多くのバグはすぐに特定できます。

もし特定のエラーやコード例があれば、それを共有してもらえれば、より具体的なデバッグ手順を案内できます。

