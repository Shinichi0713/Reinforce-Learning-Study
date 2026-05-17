
前回MAPPOを使ってAtariのboxingゲームでMARLを行いました。[1]
次なるMARLの課題について"Combat: Tank"を選定しました。

[1] 前回のMAPPO記事も御覧ください。

https://yoshishinnze.hatenablog.com/entry/2026/05/17/043000

## ゲーム概要
「Combat: Tank」は、Atari 2600用の対戦ゲーム『Combat』に含まれる**タンク対戦モード**です。  
PettingZooではこれをマルチエージェント環境として提供しており、以下のような特徴があります。

### ゲームの目的
- 2プレイヤー（2エージェント）がタンクを操作し、相手にミサイルを当てて得点を競うゲームです。
- 相手にミサイルを1発当てるごとに**+1点**、当てられた側は**-1点**が入ります[PettingZoo Documentation](https://pettingzoo.farama.org/environments/atari/combat_tank/)。
- 一定時間経過でゲーム終了となり、得点の多い方が勝ちです。

### マップと設定
- **Open Field（開けたフィールド）**と**Maze（迷路状のマップ）**の2種類のマップが用意されています。
  - PettingZooでは `has_maze` パラメータで切り替え可能です[PettingZoo Documentation](https://pettingzoo.farama.org/environments/atari/combat_tank/)。
- マップには壁や障害物があり、それらを避けながら相手を狙います。

### ミサイルの種類
オリジナルのAtari版では、以下のようなバリエーションがあります[Atari.com](https://atari.com/pages/combat)。
- **Guided Missile（誘導ミサイル）**：発射後にジョイスティックで左右に曲げて相手を追尾できる。
- **Straight Missile（直進ミサイル）**：まっすぐ飛ぶだけ。

PettingZoo環境では、これらのうちどのモードを採用しているかは環境設定に依存しますが、いずれにしても「相手に当てる」という基本ルールは同じです。

### プレイの特徴
- タンクは前進・旋回して移動し、ミサイルを発射して相手を狙います。
- 相手に当たると相手が吹き飛び、その反動で有利・不利な位置に移動することがあります[PettingZoo Documentation](https://pettingzoo.farama.org/environments/atari/combat_tank/)。
- マルチエージェント強化学習の観点では、
  - 完全対立（ゼロサム）の報酬設計（自分が+1なら相手は-1）
  - 相手の動きを読む**予測能力**と、壁を利用した**ポジショニング**が重要
という点で、典型的な対戦型MARLタスクとしてよく使われます。

まとめると、**Combat: Tankは「2台のタンクが障害物のあるマップ上でミサイルを撃ち合い、相手に当てた回数を競う対戦ゲーム」** です。  
PettingZooではこれをマルチエージェント環境として利用できるよう、報酬や観測・行動空間を整理して提供しています。



