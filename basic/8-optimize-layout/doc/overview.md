## トライアル

### reference8
dqnでレイアウトする長方形にランダム性を持たせてトライ。
学習がまったく安定しない。

### reference9
dqnのネットワーク構造を、もう少し大きくなるように修正してリトライ。
学習がまったく安定しない。
ネットワークが問題ではない・・・？

### reference10
DDQNで実装。
学習は安定するようになり、隙間を埋めて、高得点を狙えるようになった。

**成功につながったと考えるポイント**  
1. ddqnにより適正な行動評価がされるようになった。
2. ddqnは離散制御と相性が良いとされていて、今回タスク(離散空間へのレイアウト)とマッチした。

### reference11
10で高得点がとれるエージェントができたが、結構な頻度でペナルティを選択していた。
そのため、探索空間を十分にとれる、改善がいきなるではなく、段階的にされるという点に留意してリストラクトした。
![alt text](image-7.png)

**成功につながったと考えるポイント**  
1. ターゲットネットワークのパラメータ更新にソフトアップデートを取り入れて、じわじわと改善がされた

### reference12
学習フレームワークをSACに変更。
なぜか・・・。学習が安定しない。
もしかしたら、エントロピー高い領域への十分な探索を重視しすぎてるせい？

reference11で十分な結果が得られたため、一度保留。

![alt text](image-8.png)

### reference13
並べる長方形のランダム性を持たせたまま。
今度は一度並べると、長方形がなくなるように変更して学習する。

また報酬設計に、全て並べたら追加の報酬を与えるように定義する。
(というより、残りの長方形が少ない程、報酬を増やすように設計する)

__結果__  
ダメっぽい。
特に、-2→サイズ0を選択してしまっている。
__仮説__  
1. AIからして出力選択の難易度が高すぎる
2. 学習時に、サイズ0を選択することがダメと分かりづらくなっている

1は本当に分かりづらい。
修正効くなら修正してみる。

__修正点__  
1. トライのstep数を増加して、全てやりきるエピソードをとりやすくした
2. ロスを、箱のピックアップと、箱の並べるの２つで分割した(これがあるべき)
3. モデルに引き渡す情報に、直前選択した箱の番号も入れる

![alt text](image-9.png)

## reference14
学習アルゴリズムをSACに変更。

だいぶ学習の効率は改善したが、最後一押しがやはり出来てない

## reference15
モデルの構造が良くないことに気づいた

__もともと__  
1. 現状のレイアウトの状態と、箱の情報を入力後、まとめてCNNと全結合層で順電波


```python
class ActorNet(nn.Module):
    def __init__(self, size_grid, max_rects=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.rect_encoder = nn.Sequential(
            nn.Linear(max_rects * 2 + 3, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 64), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE + 64, 512), nn.ReLU()
        )
        self.box_head = nn.Linear(512, max_rects)
        self.place_head = nn.Linear(512, size_grid)

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(dir_current, "model_actor.pth")

    def forward(self, grid, rects_info):
        grid = grid.to(DEVICE)
        rects_info = rects_info.to(DEVICE)
        grid_feat = self.conv(grid)
        rect_feat = self.rect_encoder(rects_info)
        x = torch.cat([grid_feat, rect_feat], dim=1)
        x = self.fc(x)
        box_logits = self.box_head(x)
        place_logits = self.place_head(x)
        box_probs = torch.softmax(box_logits, dim=1)
        place_probs = torch.softmax(place_logits, dim=1)
        return box_probs, place_probs
```
