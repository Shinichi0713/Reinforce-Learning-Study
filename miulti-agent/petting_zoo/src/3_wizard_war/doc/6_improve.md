
前回学習コードまで作りました。そして実際学習させてみた状態でプレー動画見ると。。。

<img src="image/6_improve/wizard_2player_marl.gif" alt="q-learn" width="300px" height="auto">

やっぱりいけてないすね。。。
今回は報酬体系の改善に取り組みます。


## リファレンス
報酬の設計変更にあたり、現状把握が必要です。
Wizard of Wor（Atari 2600）の公式リファレンスは、Atari 2600 の取扱説明書や MAME の ROM 情報に基づいています。以下に、行動（操作）と報酬（スコア）の仕様を整理します。

### 1. 行動（操作）の仕様

Wizard of Wor は 2 プレイヤー協力型の迷路シューティングゲームです。各プレイヤーは以下の操作が可能です。

__基本操作__

- **移動**: 4 方向（上・下・左・右）
- **射撃**: 現在向いている方向に弾を発射
- **ワープ**: 特定の場所でワープ（マップ間移動）

__Atari 2600 版の具体的な操作__

- **ジョイスティック**: 8 方向（斜め含む）で移動、中央ボタンで射撃
- **キーボード（エミュレータ）**: 矢印キーで移動、スペースキーで射撃

__PettingZoo の `WizardOfWor-v3` における行動空間__
PettingZoo の `WizardOfWor-v3` では、行動空間は **離散的な整数** で表現されます。典型的な行動空間は以下の通りです。

- `0`: 何もしない（NOOP）
- `1`: 上に移動
- `2`: 右に移動
- `3`: 下に移動
- `4`: 左に移動
- `5`: 上に射撃
- `6`: 右に射撃
- `7`: 下に射撃
- `8`: 左に射撃
- `9`: 上右に移動（斜め）
- `10`: 右下に移動
- `11`: 左下に移動
- `12`: 左上に移動
- `13`: 上に射撃（重複の場合あり）
- `14`: 右に射撃（重複の場合あり）
- `15`: 下に射撃（重複の場合あり）
- `16`: 左に射撃（重複の場合あり）
- `17`: ワープ（特定の場所で有効）

※ 実際の行動数は環境のバージョンや設定によって異なります。`env.action_space(agent).n` で確認してください。

### 2. 報酬（スコア）の仕様

Wizard of Wor の報酬は、敵を倒した際のスコアに基づきます。公式リファレンス（Atari 2600 マニュアル）では、以下のようなスコア体系が記載されています。

__敵の種類とスコア（例）__

- **Worluk（赤い敵）**: 100 点
- **Garwor（青い敵）**: 200 点
- **Burwor（緑の敵）**: 300 点
- **Wizard of Wor（ボス）**: 1000 点

__その他の報酬__

- **ボーナス**: 特定のアイテムや条件を満たすとボーナススコア
- **残機**: 特定のスコアに達するとエクストラライフ（残機増加）

__PettingZoo の `WizardOfWor-v3` における報酬__
PettingZoo の環境では、報酬は **スコアの変化量** として与えられます。

- **基本報酬**: 敵を倒した際のスコア増加（例：100, 200, 300, 1000）
- **負の報酬**: プレイヤーが死亡した場合のペナルティ（例：-1）
- **終了報酬**: エピソード終了時の合計スコア

__報酬のスケーリング__
Atari 環境では報酬が大きくなりがちなため、学習時にスケーリングすることが一般的です。

```python
rewards = torch.FloatTensor([[rewards_dict[agent] * 0.01] for agent in env.agents])
```

### 3. 公式リファレンスの参照先

- **Atari 2600 マニュアル**: [AtariAge - Wizard of Wor Manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=598)
- **MAME ROM 情報**: [MAME - Wizard of Wor](https://www.mamedb.com/game/wizardofwor)
- **PettingZoo ドキュメント**: [PettingZoo - WizardOfWor](https://www.pettingzoo.ml/atari/wizard_of_wor)

## 報酬改良

### 改良案検討

その後でもプレーさせてみて目立った点が以下の2点でした。

1. プレーヤが敵に突っ込んでいく
2. 味方を見つけると弾を積極的に打つ

この問題ですが、以下の点が原因と考えました。
1. 敵を倒したときの報酬に比べてプレーヤーが死亡するペナルティが低すぎる
2. 味方を撃った時に報酬が激増していました。味方に対して撃破しても報酬が得られるということです

このため、報酬の改良を検討します。


### 設計変更

以下、**実際に行った報酬の改良内容**を整理します。

解決したい課題は以下と定義しました。
- **敵を倒したときの報酬**に比べて、**プレーヤーが死亡するペナルティが小さすぎる**ため、「敵に突っ込んででも敵を倒す」方が得だという学習が進んでしまっていました。

__実際に行った改良__

__(1) 死亡ペナルティの大幅強化__

`compute_rewards` 関数内で、**死亡時のペナルティを大きく負の値**に設定しました。

```python
# 死亡ペナルティ
death_penalty = 0.0
if terminations.get(agent, False) or truncations.get(agent, False):
    death_penalty = -50.0  # 以前より大幅に大きく
```

これにより、
- 敵を倒しても `+1` 程度の報酬
- 自分が死ぬと `-50` の報酬

というバランスになり、**「無理に突っ込んで敵を倒す」よりも「生き残る」方が得**という学習が進むようになりました。

__(2) 生存報酬の追加__

さらに、**生きているだけで小さな正の報酬**を与えるようにしました。

```python
# 生存時間報酬（敵弾回避の近似）
survival_reward = 0.1
```

これにより、
- 敵を倒さなくても、**生きているだけで報酬が溜まる**
- 無理な突撃で死ぬより、**安全に生き残る方が得**

という方向に学習が進むようになりました。

__2. 味方を見つけると積極的に撃つ問題への対応__

__原因__

- **味方を撃ったときにも「敵撃破」と同じ報酬が入っていた**ため、「敵を探すより味方を撃った方が確実に報酬が得られる」という学習が進んでしまっていました。

__実際に行った改良__

__(1) 味方撃ちペナルティの導入__

`compute_rewards` 関数内で、**「味方を撃った」とみなされる行動に対して大きな負の報酬**を追加しました。

```python
# ★ 味方撃ちペナルティ
friendly_fire_penalty = 0.0
if (terminations.get(agent, False) or truncations.get(agent, False)) and shot_this_step[agent]:
    # 死んだ かつ 直前で弾を撃った → 味方を撃った可能性が高い
    friendly_fire_penalty = -50.0  # 味方撃ちペナルティ
```

ここで `shot_this_step[agent]` は、**直前のステップでそのエージェントが弾を撃ったかどうか**を記録した辞書です。

これにより、
- 味方を撃って倒しても `+1` 程度の報酬
- 味方撃ちペナルティで `-50` の報酬

となり、**味方を撃つと大きく損をする**ため、味方を狙う行動が抑制されます。

__(2) 弾発射の記録ロジックの追加__

学習ループ内で、**「弾を撃ったかどうか」** を記録する仕組みを追加しました。

```python
# 弾発射の記録用
shot_this_step = {agent: False for agent in env.possible_agents}

# ...

with torch.no_grad():
    value, action, action_log_prob = ac.get_actions(
        cur_obs.to(device),
        cur_state.to(device),
        agent_id_onehot=agent_ids_onehot
    )

# ★ 弾発射の記録
for i, agent in enumerate(env.agents):
    if action[i].item() == 1:  # 例: 行動1が「弾を撃つ」
        shot_this_step[agent] = True
    else:
        shot_this_step[agent] = False
```

この `shot_this_step` を `compute_rewards` に渡すことで、**「弾を撃った直後に死んだ」**という状況を「味方撃ち」とみなしてペナルティを課しています。

## 学習した結果

上記の設計変更後に再度学習させてみました。
学習後の動画が以下です。

<img src="image/6_improve/episode_20260529_235615.gif" alt="q-learn" width="300px" height="auto">

敵が近づいてくるとプレーヤは弾を撃ちながら進んでいきます。
むやみに敵に突っ込むような行動はとらなくなりました。

## 総括

初期の状態学習させたプレイ動画を確認したところ、プレイヤーが敵に無謀に突っ込んで死亡したり、味方を積極的に撃つといった不自然な行動が目立ちました。  
これは、敵を倒したときの報酬に比べて死亡ペナルティが小さすぎること、また味方を撃破しても敵撃破と同じ報酬が入ってしまうことが原因かしらと思われます。

そこで報酬設計をデフォルトの状態より改良し、死亡時に大きな負の報酬（-50）を与えるとともに、生存しているだけで小さな正の報酬（+0.1）を追加しました。  
さらに、弾を撃った直後に死亡した場合を「味方撃ち」とみなして大きなペナルティ（-50）を課す仕組みを導入しました。

改良後の学習の結果、敵に近づきつつ安全に弾を撃つ、より自然なプレイが見られるようになりました。

やはり報酬の改良もセットですね。。。

<div class="shop-card">
<div class="shop-card-image"><img src="https://m.media-amazon.com/images/I/81lem2peqFL._SL1500_.jpg" alt="商品画像" /></div>
<div class="shop-card-content">
<div class="shop-card-title">強化学習 (機械学習プロフェッショナルシリーズ)</div>
<div class="shop-card-description">同シリーズで緑本のPythonによる強化学習の本を何度も何度も読んだのですが、どうしても読み進めません。試しにと思って3年前に買ったこの本を読み返してみるとすっと読めました。 これからのコーディングは生成AIが書いてくれるのだから、難しい理論本で勉強してコーディングはお任せ（直すべき所は直す）というのが正解なのかもしれない。。。</div>
<div class="shop-card-link"><a href="https://www.amazon.co.jp/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E6%A3%AE%E6%9D%91%E5%93%B2%E9%83%8E-ebook/dp/B07XJXMQGD?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&amp;crid=2Q7JANDTXMDRQ&amp;dib=eyJ2IjoiMSJ9.YZxuAtwvMTmksETM7b4V5tEFcZKwS3FH_fG2YEbWKvrGjHj071QN20LucGBJIEps.GCkT5rik7rfwPmJpLUkBFsUfiUvfOc-QO8WH5HT0oSA&amp;dib_tag=se&amp;keywords=MARL+%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92&amp;qid=1777879215&amp;sprefix=marl+%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%2Caps%2C165&amp;sr=8-1&amp;linkCode=ll2&amp;tag=yoshishinnze-22&amp;linkId=a3ac27efe00549a8b95a7d948fa658b0&amp;ref_=as_li_ss_tl" target="_blank" rel="noopener">Amazonで詳細を見る</a></div>
</div>
</div>
<p>[blog:g:4207112889963697807:banner]</p>
<p>[blog:g:10328749687175353006:banner]</p>
<p>[blog:g:11696248318754550880:banner]</p>
<p>[blog:g:11696248318754550877:banner]</p>

