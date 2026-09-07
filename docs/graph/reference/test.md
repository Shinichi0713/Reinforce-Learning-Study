__BFS例題の解説__

コードの実行結果をもとに、**なぜ訪問順が A → B → C → D になるのか**を詳しく説明します。

__1. このグラフの構造__

```
     A
    / \
   B   C
    \ /
     D
```

| 頂点 | 隣接する頂点 | 意味 |
|------|-------------|------|
| A | B, C | AからBとCに直接行ける |
| B | A, D | BからAに戻れて、Dにも行ける |
| C | A, D | CからAに戻れて、Dにも行ける |
| D | B, C | DからBとCに戻れる |

__2. ステップごとの動作__

**ステップ1：Aを処理**

| 項目 | 内容 |
|------|------|
| キューの状態 | [A] |
| 取り出す頂点 | **A** |
| Aの隣接 | B, C |
| キューに追加 | B, C |

Aを訪問し、隣接するBとCをキューに追加します。
**重要：Bが先に追加される**ため、BがCより先に処理されます。

**ステップ2：Bを処理**

| 項目 | 内容 |
|------|------|
| キューの状態 | [B, C] |
| 取り出す頂点 | **B**（先に入れたものが先に出る = FIFO） |
| Bの隣接 | A, D |
| キューに追加 | **D**（Aは訪問済みなので無視） |

Bを訪問し、隣接するAは「訪問済み」なので無視し、**Dをキューに追加**します。

**ステップ3：Cを処理**

| 項目 | 内容 |
|------|------|
| キューの状態 | [C, D] |
| 取り出す頂点 | **C** |
| Cの隣接 | A, D |
| キューに追加 | **なし**（AもDも訪問済み） |

Cを訪問しますが、Aは訪問済み、Dはステップ2で追加済みなので、**新しい頂点は追加されません**。

**ステップ4：Dを処理**

| 項目 | 内容 |
|------|------|
| キューの状態 | [D] |
| 取り出す頂点 | **D** |
| Dの隣接 | B, C |
| キューに追加 | **なし**（両方とも訪問済み） |

Dを訪問しますが、BとCはすでに訪問済みなので、新しい頂点は追加されません。
キューが空になり、**探索終了**です。

__3. なぜこの順序になるのか？__

__核心：「距離順」に処理される__

| 頂点 | 始点Aからの距離 | 処理ステップ |
|------|---------------|-------------|
| A | 0 | ステップ1 |
| B | 1 | ステップ2 |
| C | 1 | ステップ3 |
| D | 2 | ステップ4 |

- **距離0**：Aが最初に処理される
- **距離1**：BとCが次に処理される（Aの隣接）
- **距離2**：Dが最後に処理される（BまたはCの隣接）

__なぜBがCより先に処理されるのか？__

Aの隣接リストが `['B', 'C']` の順なので、**Bが先にキューに入ります**。
キューはFIFO（先入れ先出し）なので、Bが先に出てきます。

> もし `['C', 'B']` の順なら、訪問順は **A → C → B → D** になります。最短距離自体は変わりません。

__4. 「訪問済みチェック」の重要性__

各ステップで `if neighbor not in seen:` を確認しています。

- **Aの処理時**：B, Cを追加
- **Bの処理時**：Aは訪問済みなので無視、Dを追加
- **Cの処理時**：Aは訪問済みなので無視、Dは追加済みなので無視
- **Dの処理時**：B, Cともに訪問済みなので無視

このチェックがないと、**A↔B↔D↔C↔A と無限ループ**に陥る可能性があります。

__5. BFSの「波紋」のイメージ__

```
ステップ1:  [A]           ← 距離0
ステップ2:  [B, C]        ← 距離1（Aの隣接）
ステップ3:  [C, D]        ← Cは距離1、Dは距離2
ステップ4:  [D]           ← 距離2
```

まさに**池の波紋**のように、始点Aから同心円状に探索が広がっていきます。

__確認の質問__

以下のグラフで、始点を1としてBFSを実行した場合、**訪問順**は何になるでしょうか？

```
    1
   / \
  2   3
  |   |
  4   5
   \ /
    6
```

（隣接リスト：`{1: [2,3], 2: [1,4], 3: [1,5], 4: [2,6], 5: [3,6], 6: [4,5]}` と仮定してください）



DFS（深さ優先探索）の例題と体験用コードをまとめて提供いたします。

---

## 例題1：グラフ版DFS

### グラフ構造

```
        A
       / \
      B   C
      |   |
      D   E
       \ /
        F
```

### 隣接リスト

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'F'],
    'E': ['C', 'F'],
    'F': ['D', 'E']
}
```

### DFS訪問順

**A -> B -> D -> F -> E -> C**

### 動作のポイント

1. Aを訪問 → 隣接は[B, C] → Bを選んで奥へ進む（Cは後回し）
2. Bを訪問 → 隣接は[A, D] → Aは訪問済み → Dを選んで奥へ進む
3. Dを訪問 → 隣接は[B, F] → Bは訪問済み → Fを選んで奥へ進む
4. Fを訪問 → 隣接は[D, E] → Dは訪問済み → Eを選んで奥へ進む
5. Eを訪問 → 隣接は[C, F] → Fは訪問済み → Cを選んで奥へ進む
6. Cを訪問 → 隣接は[A, E] → 両方訪問済み → 行き止まり → バックトラック

---

## 例題2：グリッド迷路版DFS

### 迷路

```
  0 1 2 3 4
0 S . . # .
1 # # . # .
2 . . . . .
3 . # # # .
4 . . . . G
```

### DFSで発見した経路

**(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4) -> (3,4) -> (4,4)**

経路の長さ: 9

---

## 例題3：サイクル検出（DFSの応用）

### 問題

以下のグラフにサイクル（閉路）があるかどうか、DFSを使って判定する。

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
```

### 解答の核心

```python
def has_cycle_dfs(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node, parent):
        color[node] = GRAY  # 探索中
        
        for neighbor in graph[node]:
            if color[neighbor] == GRAY and neighbor != parent:
                return True  # サイクル発見
            if color[neighbor] == WHITE:
                if dfs(neighbor, node):
                    return True
        
        color[node] = BLACK  # 探索完了
        return False
    
    for node in graph:
        if color[node] == WHITE:
            if dfs(node, None):
                return True
    return False
```

**結果**: True（サイクルあり）

---

## 例題4：トポロジカルソート（DFSの応用）

### 問題

大学の授業で以下の前提条件がある。DFSを使って履修可能な順序を求める。

- 「微積分」を履修してから「力学」が取れる
- 「微積分」を履修してから「電磁気学」が取れる
- 「力学」を履修してから「解析力学」が取れる
- 「電磁気学」を履修してから「量子力学」が取れる

### 解答の核心

```python
def topological_sort_dfs(graph):
    visited = set()
    order = []
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)  # ポストオーダー
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    order.reverse()
    return order

courses = {
    '微積分': ['力学', '電磁気学'],
    '力学': ['解析力学'],
    '電磁気学': ['量子力学'],
    '解析力学': [],
    '量子力学': []
}
```

**結果**: 微積分 -> 電磁気学 -> 量子力学 -> 力学 -> 解析力学

---

## 例題5：連結成分の列挙（DFSの応用）

### 問題

以下のグラフにいくつの独立したグループ（連結成分）があるか、DFSを使って列挙する。

```
  グループ1      グループ2      グループ3
    A---B          C---D          E
    |
    F
```

### 解答の核心

```python
def find_connected_components(graph):
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components
```

**結果**: 3つの連結成分
- グループ1: ['A', 'B', 'F']
- グループ2: ['C', 'D', 'G']
- グループ3: ['E']

---

## 例題6：バックトラックで数独を解く（DFSの応用）

### 問題

以下の簡易数独（4x4）をDFS（バックトラック）で解く。

```
  1 0 | 0 4
  0 2 | 3 0
  ----+----
  0 3 | 0 1
  4 0 | 2 0
```

### 解答の核心

```python
def solve_sudoku_dfs(board):
    def is_valid(board, row, col, num):
        # 行チェック
        for c in range(4):
            if board[row][c] == num:
                return False
        # 列チェック
        for r in range(4):
            if board[r][col] == num:
                return False
        # 2x2ブロックチェック
        block_row, block_col = 2 * (row // 2), 2 * (col // 2)
        for r in range(block_row, block_row + 2):
            for c in range(block_col, block_col + 2):
                if board[r][c] == num:
                    return False
        return True
    
    def dfs():
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    for num in range(1, 5):
                        if is_valid(board, row, col, num):
                            board[row][col] = num  # 置いてみる
                            if dfs():  # 奥へ進む
                                return True
                            board[row][col] = 0  # バックトラック
                    return False
        return True
    
    dfs()
    return board
```

---

## DFSの核心まとめ

| 特徴 | 説明 |
|------|------|
| **探索戦略** | 奥へ奥へ突き進み、行き止まりで戻る（バックトラック） |
| **データ構造** | スタック（LIFO）または再帰 |
| **用途1** | サイクル検出 |
| **用途2** | トポロジカルソート |
| **用途3** | 連結成分の列挙 |
| **用途4** | バックトラックによる解探索（数独、迷路など） |
| **注意点** | 最短経路は保証されない |

---

## BFSとの比較

| 比較項目 | BFS | DFS |
|----------|-----|-----|
| **データ構造** | キュー（FIFO） | スタック（LIFO）または再帰 |
| **探索順** | 近い順に同心円状 | 奥へ奥へ突き進む |
| **最短ステップ数** | 保証される | 保証されない |
| **サイクル検出** | 不向き | 得意 |
| **トポロジカルソート** | 不可能 | 得意 |
| **メモリ使用量** | 幅が広いと多い | 深さが浅ければ少ない |

---

## 体験用コード

### dfs_exercise.py

```python
from collections import deque

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    order = []
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        order.append(node)
        
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return order

def dfs_recursive(graph, start):
    visited = set()
    order = []
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            dfs(neighbor)
    
    dfs(start)
    return order

if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    print("反復版DFS:", " -> ".join(dfs_iterative(graph, 'A')))
    print("再帰版DFS:", " -> ".join(dfs_recursive(graph, 'A')))
```

---

## 確認の質問

以下のグラフで、始点を1としてDFSを実行した場合、**訪問順**は何になるでしょうか？

```
    1
   / \
  2   3
 /     \
4       5
```

隣接リスト: `{1: [2, 3], 2: [1, 4], 3: [1, 5], 4: [2], 5: [3]}`