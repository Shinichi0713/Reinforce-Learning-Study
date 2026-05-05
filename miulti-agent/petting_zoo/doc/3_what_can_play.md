PettingZooのAtari環境ライブラリには、マルチエージェント（対戦・協力）に対応したAtari 2600のゲームが多数含まれています。

これらはArcade Learning Environment (ALE) をベースにしており、以下のような主要なタイトルが提供されています。

### 主なAtari環境リスト

PettingZooでサポートされている代表的なマルチエージェント対応タイトルは以下の通りです。

* **対戦型 (Competitive)**
  * `boxing_v2`: ボクシング（2人対戦）
  * `pong_v3`: ポン（2人対戦）
  * `tennis_v3`: テニス（2人対戦）
  * `surround_v2`: 陣取りゲーム（2人対戦）
  * `ice_hockey_v2`: アイスホッケー（2人対戦）
  * `entombed_competitive_v3`: 迷路脱出（2人対戦）
* **協力型 (Cooperative)**
  * `entombed_cooperative_v3`: 協力して迷路を進む（2人協力）
  * `space_invaders_v2`: スペースインベーダー（2人協力）
  * `wizard_of_wor_v3`: ウィザード・オブ・ウォー（2人協力）
* **その他（多人数・特殊）**
  * `mario_bros_v3`: マリオブラザーズ
  * `joust_v3`: ジャウスト
  * `warlords_v3`: ウォーローズ（4人対戦）
  * `quadrapong_v4`: クアドラポン（4人対戦）
  * `flag_capture_v2`: 旗取りゲーム

---

### 利用方法と注意点

#### 1. インストール

Atari環境を利用するには、本体とは別に `atari`用の依存関係とROMが必要です。

**Bash**

```
pip install 'pettingzoo[atari]'
# ROMのインストールが必要な場合があります
pip install autorom
AutoROM --accept-gpl-license
```

#### 2. 基本的なインポート

例えば、`boxing`環境を使用する場合は以下のようにインポートします。

**Python**

```
from pettingzoo.atari import boxing_v2

env = boxing_v2.env()
env.reset()
```

#### 3. 推奨される前処理 (SuperSuit)

Atari環境はそのままでは学習が難しいため、`SuperSuit`というライブラリを併用して、フレームのリサイズやグレースケール化、フレームスタッキングを行うのが一般的です。

* `resize_v1`: 解像度の変更（例: 84x84）
* `color_reduction_v0`: グレースケール化
* `frame_stack_v1`: 過去のフレームを重ねて動きを捉えやすくする
* `sticky_actions_v0`: 決定的すぎる挙動を防ぐためのランダム性追加

PettingZooのAtari環境は、単一エージェントのGym (Gymnasium) 版とは異なり、エージェントごとの報酬や観測が管理されるため、MARL（マルチエージェント強化学習）のアルゴリズム検証に非常に適しています。
