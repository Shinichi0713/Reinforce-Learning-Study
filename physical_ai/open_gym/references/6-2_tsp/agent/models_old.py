import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.distributions import Categorical


class Encoder(object):

    def __init__(self, n_neurons=128, batch_size=4,
                 seq_length=10):
        # パラメタ設定
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 再帰セル定義
        self.enc_rec_cell = LSTMCell(self.n_neurons)

    # ネットワーク定義
    # Decoderとの対比から、(LSTMレイヤでなく)敢えて明示的に
    # Loopで記載
    def build_model(self, inputs):

        # Bi-directional LSTMレイヤを挿入
        inputs = Bidirectional(LSTM(self.n_neurons,
                               return_sequences=True),
                               merge_mode='concat')(inputs)

        input_list = tf.transpose(inputs, [1, 0, 2])

        enc_outputs, enc_states = [], []
        state = self._get_initial_state()

        for input in tf.unstack(input_list, axis=0):
            # 再帰ネットワークへの入出力
            output, state = self.enc_rec_cell(
                input, state)

            enc_outputs.append(output)
            enc_states.append(state)

        # 出力の蓄積
        enc_outputs = tf.stack(enc_outputs, axis=0)
        enc_outputs = tf.transpose(enc_outputs,
                                   [1, 0, 2])

        enc_state = enc_states[-1]

        return enc_outputs, enc_state

    def _get_initial_state(self):

        state = self.enc_rec_cell.get_initial_state(inputs=None,
                                                    batch_size=self.batch_size,
                                                    dtype=tf.float32)
        return state


class ActorDecoder(object):

    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        # パラメタ設定
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 地点マスク用罰則
        self.infty = 1.0E+08
        # 地点マスクbit（テンソル）
        self.mask = 0
        # サンプリングのシード
        self.seed = None

        # 初期入力値のパラメータ変数(Encoderセルの出力次元、
        # [batch_size, n_neuron])
        first_input = tf.get_variable(
            'GO', [1, self.n_neurons])
        self.dec_first_input = tf.tile(
            first_input, [self.batch_size, 1])

        # Pointing機構のパラメータ変数
        self.W_ref = tf.get_variable(
            'W_ref',
            [1, self.n_neurons, self.n_neurons])
        self.W_out = tf.get_variable(
            'W_out',
            [self.n_neurons, self.n_neurons])
        self.v = tf.get_variable('v', [self.n_neurons])

        # 再帰セル定義
        self.dec_rec_cell = LSTMCell(self.n_neurons)

    def set_seed(self, seed):
        self.seed = seed

    # ネットワーク定義
    # Pointing機構による出力列(ネットワーク)の構成と、
    # 対応対数尤度の算出
    def build_model(self, enc_outputs, enc_state):

        output_list = tf.transpose(enc_outputs,
                                   [1, 0, 2])

        locations, log_probs = [], []

        input, state = self.dec_first_input, enc_state
        for step in range(self.seq_length):

            # 再帰ネットワークへの入出力
            output, state = self.dec_rec_cell(
                input, state)

            # Pointing機構への入出力
            masked_scores = self._pointing(
                enc_outputs, output)

            # 各入力地点の選択(logit)スコアを持った多項分布の定義
            prob = Categorical(logits=masked_scores)

            # 確率に応じた次地点の選択と、該当対数尤度の定義
            location = prob.sample(seed=self.seed)
            # 選択地点の登録
            locations.append(location)

            # 選択地点の対数尤度(テンソル)の算出
            logp = prob.log_prob(location)
            # 対数尤度の登録
            log_probs.append(logp)

            # 既訪問地点マスクと次入力の更新
            self.mask = self.mask + tf.one_hot(
                location, self.seq_length)
            input = tf.gather(output_list, location)[0]

        # 初期地点の再追加（距離/報酬算出の利便性のため）
        first_location = locations[0]
        locations.append(first_location)

        tour = tf.stack(locations, axis=1)
        log_prob = tf.add_n(log_probs)

        return log_prob, tour

    # Pointing機構定義
    # Encoder出力群(のEmbedding)の情報+Decoder出力の
    # 逐次情報から、参照先別の(logit)スコアを算出
    def _pointing(self, enc_outputs, dec_output):

        # Encoder出力項([batch_size, seq_length, n_neuron])
        enc_term = tf.nn.conv1d(enc_outputs, self.W_ref,
                                1, 'VALID')

        # Decoder出力項([batch_size, 1, n_neuron])
        dec_term = tf.expand_dims(
            tf.matmul(dec_output, self.W_out), 1)

        # 参照先別のスコアの算出([batch_size, seq_length])
        scores = tf.reduce_sum(
            self.v * tf.tanh(enc_term + dec_term), [-1])

        # 既訪問地点(batchごとに異なる)のスコアに-inftyを付与
        masked_scores = scores - self.infty * self.mask

        return masked_scores


class CriticDecoder(object):

    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        # パラメタ設定
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length

        # glimpsing機構のネットワーク変数
        self.W_ref_g = tf.get_variable('W_ref_g',
                                       [1, self.n_neurons, self.n_neurons])
        self.W_q_g = tf.get_variable('W_q_g', [self.n_neurons, self.n_neurons])
        self.v_g = tf.get_variable('v_g', [self.n_neurons])

    # ネットワーク定義
    def build_model(self, enc_outputs, enc_state):

        # Insert glimpsing process, etc. if necessary, before Main Layer

        # Glimpsing Layer
        # [n_batch, n_neuron]
        frame = enc_state[0]

        # attention part
        # [n_batch, n_seq, n_neuron]
        enc_ref_g = tf.nn.conv1d(enc_outputs, self.W_ref_g, 1, 'VALID',
                                 name='encoded_ref_g')
        # [n_batch, 1, n_neuron]
        enc_q_g = tf.expand_dims(tf.matmul(frame, self.W_q_g,
                                 name='encoded_q_g'), 1)
        # [n_batch, n_seq]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(enc_ref_g + enc_q_g), [-1],
                                 name='scores_g')
        # [n_batch, n_seq]
        attention_g = tf.nn.softmax(scores_g, name='attention_g')

        # glimpsing part
        # linear combination of reference vectors (defines new query vector)
        # [n_batch, n_seq, n_neuron]
        glimpse = tf.multiply(enc_outputs, tf.expand_dims(attention_g, axis=2))
        # [n_batch, n_neuron]
        glimpse = tf.reduce_sum(glimpse, axis=1)

        # # No Additional Layer (for backup)
        # # simple glimpsing, [n_batch, n_seq, n_neuron] to [n_batch, n_neuron]
        # glimpse = tf.reduce_mean(enc_outputs, axis=1)

        # Main FC Layer

        hidden = Dense(self.n_neurons, activation='relu')(glimpse)
        baseline = Dense(1, activation='linear')(hidden)

        return baseline
