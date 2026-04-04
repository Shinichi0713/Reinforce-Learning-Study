


## PyMarl2とは

PyMARL2 は、**マルチエージェント強化学習（Multi-Agent Reinforcement Learning, MARL）のアルゴリズム実装とベンチマークのためのオープンソースフレームワーク**です。
PyMARLというパッケージを回収したものです。

### そもそもPyMARLとは

PyMARL は、**マルチエージェント強化学習（Multi-Agent Reinforcement Learning, MARL）のための Python フレームワーク**で、主に協調型タスクの研究・実験に使われます。

- **開発元・目的**  
  Oxford の WhiRL（Whiteson Research Lab）が開発した、**深層マルチエージェント強化学習用のフレームワーク**です[GitHub - oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)。  
  PyTorch で実装されており、MARL アルゴリズムの実装・比較・実験をしやすくすることを目的としています。

- **実装されている主なアルゴリズム**  
  代表的な協調型 MARL アルゴリズムが複数実装されています。  
  例：  
  - QMIX（Monotonic Value Function Factorisation）  
  - COMA（Counterfactual Multi-Agent Policy Gradients）  
  - VDN（Value-Decomposition Networks）  
  - IQL（Independent Q-Learning）  
  - QTRAN など[GitHub - oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)

- **対応環境**  
  主な環境として **SMAC（StarCraft Multi-Agent Challenge）** が使われます。  
  SMAC は StarCraft II をベースにしたマルチエージェント協調タスクのベンチマーク環境で、PyMARL は SMAC と組み合わせてアルゴリズムの性能評価を行うことを想定しています[GitHub - oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)。

- **主な特徴**  
  - **Docker ベースのセットアップ**：環境構築を簡単にするための Docker サポート  
  - **設定ファイル駆動の実験管理**：`src/config` による実験設定の管理  
  - **モデルの保存・読み込み（checkpointing）**：学習済みモデルの保存・再開  
  - **StarCraft II リプレイの統合**：学習結果を StarCraft II のリプレイとして可視化できる機能  
  など、研究用途に便利な機能が揃っています[GitHub - oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)。

- **研究コミュニティでの位置づけ**  
  PyMARL は SMAC の公式実装として広く利用されており、多くの MARL 論文のベースラインとしても使われています。  
  また、PyMARL を拡張した **EPyMARL（Extended PyMARL）** や、さらにそれを発展させた **PyMARL2 / PyMARL3** といった後継・拡張フレームワークも存在します[agents-lab.org - EPyMARL](https://agents-lab.org/blog/epymarl/)。

まとめると、PyMARL は  
「**SMAC 環境で QMIX などの協調 MARL アルゴリズムを実装・実験するための、研究向けの標準的なフレームワーク**」  
と理解しておくとよいでしょう。

### そしてPyMARL2とは

主な特徴は以下の通りです。

- **目的**  
  複数のエージェントが協調するタスク（例：StarCraft Multi-Agent Challenge, SMAC）で、QMIX などの代表的な MARL アルゴリズムを**公平に比較・評価**できるようにすることを目的としています。

- **PyMARL からの発展**  
  PyMARL は SMAC の著者らが公開した最初の MARL フレームワークで、PyMARL2 はその改良版・拡張版として位置づけられています。  
  具体的には、**実装上のテクニック（implementation tricks）やモジュール構造を整理し、アルゴリズムを差し替えやすくする**ことを狙っています[GitHub - hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)。

- **実装されている主なアルゴリズム**  
  代表的な協調型 MARL アルゴリズム（QMIX, QTRAN, COMA など）が実装されており、SMAC 環境での性能比較や再現実験に広く使われています。

- **研究コミュニティでの位置づけ**  
  - PyMARL2 を利用して「MARL アルゴリズムの性能比較」を行う論文も存在します[UF JUR - A Performance Comparison of MARL Algorithms using PyMARL2](https://journals.flvc.org/UFJUR/article/view/138769)。  
  - さらに、PyMARL2 を拡張して**置換不変性・置換同変性**を付与した PyMARL3 という後継フレームワークも提案されています[GitHub - tjuHaoXiaotian/pymarl3](https://github.com/tjuHaoXiaotian/pymarl3)。

まとめると、PyMARL2 は  
「**SMAC などの協調マルチエージェント環境で、複数の MARL アルゴリズムをモジュール的に実装・比較できる実験用フレームワーク**」  
と理解しておくとよいでしょう。


## PyMARL2の環境構築

PyMARL2 を Google Colab で使うには、**リポジトリをクローンして依存パッケージをインストールし、StarCraft II / SMAC 環境をセットアップ**する必要があります。

以下、Colab 用に整理した手順です。

### 1. リポジトリのクローン

Colab のセルで以下を実行します。

```bash
!git clone https://github.com/hijkzzz/pymarl2.git
%cd pymarl2
```

これで `pymarl2` ディレクトリにソースコードが入ります[GitHub - hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)。

### 2. 依存パッケージのインストール

公式のインストール手順では conda 環境を使いますが、Colab では `pip` で直接インストールするのが簡単です。

```bash
!bash install_dependecies.sh
```

このスクリプトは、PyTorch やその他の必要な Python パッケージをインストールします[GitHub - hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)。

### 3. StarCraft II / SMAC のセットアップ

SMAC 環境を使うには、StarCraft II 本体とマップデータが必要です。

```bash
!bash install_sc2.sh
```

このスクリプトは：
- StarCraft II バージョン 2.4.10 を `3rdparty` フォルダにダウンロード
- SMAC 用のマップファイルをコピー  
という処理を行います[GitHub - hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)。

### 4. Google Football 環境（オプション）

サッカー環境も使いたい場合は、別途以下を実行します。

```bash
!bash install_gfootball.sh
```

### 5. Colab 特有の注意点

- **GPU の有効化**：  
  Colab の「ランタイム」→「ランタイムのタイプ変更」で GPU を有効にしておくと、学習が速くなります。

- **セッション再開時の再インストール**：  
  Colab のランタイムを再起動すると、`/content` 以下は初期化されます。  
  そのため、**毎回 `git clone` と `install_*.sh` を実行する**必要があります。

- **StarCraft II のライセンス**：  
  SC2 は Blizzard のゲームですが、SMAC で使うための Linux 版は無料で利用できます。  
  ただし、Colab の外部ネットワークアクセスやディスク容量に注意してください。

### 6. 動作確認の例

インストールが終わったら、例えば以下のように簡単なテストを走らせてみます。

```bash
# 例：SMAC の簡単な実行テスト（README や examples を参照）
# !python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m
```














