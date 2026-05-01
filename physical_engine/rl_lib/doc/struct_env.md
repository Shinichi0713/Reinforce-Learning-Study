RLlib + AI Economist の環境構築は、**Google Colab で試す場合**と**ローカル環境で動かす場合**で分けて説明します。

---

## 1. Google Colab で動かす場合（推奨）

Colab では、Salesforce 公式のチュートリアルノートブックを使うのが一番簡単です。

### 1-1. ノートブックを開く

まず、公式チュートリアルを開きます。  
- [multi_agent_training_with_rllib.ipynb](https://colab.research.google.com/github/salesforce/ai-economist/blob/master/tutorials/multi_agent_training_with_rllib.ipynb)

### 1-2. ライブラリのインストール

ノートブック内のインストールセルに従って、以下を実行します。

```bash
# ai-economist のインストール
!pip install ai-economist

# Ray + RLlib のインストール
!pip install ray[rllib]
```

Colab のランタイムは Python 3.7+ なので、そのまま動きます。  
（ノートブック側でバージョン指定などがあれば、そちらに従ってください。）

### 1-3. 追加の依存関係（必要に応じて）

環境によっては、`tensorflow` や `torch` などが別途必要になる場合があります。  
ノートブックに記載があれば、それに従ってインストールしてください。

### 1-4. 実行

あとはノートブックのセルを上から順に実行していけば、  
AI Economist の環境を RLlib でマルチエージェント学習する流れを確認できます。

---

## 2. ローカル環境で動かす場合

### 2-1. Python のバージョン確認

AI Economist は Python 3.7+ を前提としています。  
`python --version` で 3.7 以上であることを確認してください。

### 2-2. パッケージのインストール

```bash
pip install ai-economist
pip install ray[rllib]
```

必要に応じて、`tensorflow` や `torch` もインストールします。

```bash
# TensorFlow を使う場合
pip install tensorflow

# PyTorch を使う場合（公式サイトのコマンド推奨）
# https://pytorch.org/get-started/locally/
```

### 2-3. GPU を使う場合（オプション）

GPU がある環境で RLlib の学習を高速化したい場合は、CUDA 対応の PyTorch/TensorFlow を入れておきます。  
Colab の場合はランタイムを「GPU」に切り替えるだけで済みます。

### 2-4. 動作確認

公式リポジトリのサンプルコードやチュートリアルを動かして、環境が正しく構築されているか確認します。

- AI Economist リポジトリ: [salesforce/ai-economist](https://github.com/salesforce/ai-economist)

---

## 3. よくある注意点

- **Colab のランタイム制限**  
  長時間学習する場合は、Colab のセッションが切れないように注意してください。  
  ローカルで本格的に学習する場合は、GPU 付きマシンやクラウドインスタンスを検討するのが現実的です。

- **Ray / RLlib のバージョン互換**  
  `ai-economist` と `ray[rllib]` のバージョンが合わないとエラーが出ることがあります。  
  公式チュートリアルで使われているバージョンに合わせるのが安全です。

- **Python 3.10+ での注意**  
  一部の依存ライブラリが最新 Python でうまく動かない場合があります。  
  その場合は Python 3.8 や 3.9 を使うと安定しやすいです。

---

## まとめ

- **Colab で手軽に試す** → 公式ノートブックを開き、`pip install ai-economist ray[rllib]` して実行するだけです。  
- **ローカルで本格的に使う** → Python 3.7+ 環境に同じパッケージを入れ、GPU 設定なども調整します。

まずは Colab の公式チュートブルで動かしてみて、動作確認してからローカルに移行するのがおすすめです。