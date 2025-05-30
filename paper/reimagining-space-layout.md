# 再イメージング状態レイアウトデザインwith DRL

SLDは中心的な役割を果たす。

レーザーウォールが直進の壁か曲がった壁であるべき。
フロアプランに壁の基礎が形作られた時に、セグメントから放たれる光が既存の壁がソフトパートまでか、ハードパート。

## 状態空間
RLはエージェントの行動背景となる。
累積的な報酬を最大化することを狙って、エージェントは行動を行う。
状態空間は、２つの観点で感知される。エージェントの観点と、環境の観点です。

このようにして、エージェントと環境は、意思決定するための状態の知識が必要となる。
スペースレイアウトは、離散セルの２次元により構成される。
初めに、画像の特徴量を通して、NNに通す前にグレースケースに加工されている。
次に、特徴量ベクトルを通して、どの壁も、そのほかの壁も


## 実験

CNNのレイヤを熱くした状態で、繰り返し学習した場合に報酬がどのように推移するか確認する。

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),
    nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
    nn.Flatten()
)
```

```
Episode 0: Total reward = 13.0
Episode 1: Total reward = 20.0
Episode 2: Total reward = 12.0
Episode 3: Total reward = 38.0
Episode 4: Total reward = 16.0
```

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
    nn.Flatten()
)
```

```
Episode 0: Total reward = 23.0
Episode 1: Total reward = 30.0
Episode 2: Total reward = 10.0
Episode 3: Total reward = 11.0
Episode 4: Total reward = 17.0
```

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),          # 3x3: 基本的な特徴抽出
    nn.Conv2d(16, 32, 1, stride=1), nn.ReLU(),         # 1x1: チャネル変換
    nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.ReLU(), # 5x5: 広い特徴抽出（パディングでサイズ調整）
    nn.Conv2d(32, 32, 1, stride=1), nn.ReLU(),         # 1x1: チャネル変換
    nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), # 3x3: 追加の特徴抽出
    nn.Flatten()
)
```


```
Episode 0: Total reward = 19.0
Episode 1: Total reward = 41.0
Episode 2: Total reward = 20.0
Episode 3: Total reward = 40.0
Episode 4: Total reward = 19.0
```

CNNのレイヤのサイズのせいではない。  
入力画像のサイズを変更して、変化が出るかを確認する。



```python
```

```
Episode 0: Total reward = 12.0
Episode 1: Total reward = 10.0
Episode 2: Total reward = 14.0
Episode 3: Total reward = 41.0
Episode 4: Total reward = 20.0
```

