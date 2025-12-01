import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
import numpy as np


# Critic クラスの定義
class Critic:

    def __init__(self, num_states, learning_rate=1e-3):
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.model = self._build_network()
        self._compile_graph(self.model)

    # Critic のニューラルネットワーク表現を関数として定義
    def _build_network(self):
        num_dense_1 = self.num_states * 10
        num_dense_3 = 5
        num_dense_2 = int(
            np.sqrt(num_dense_1 * num_dense_3))

        l_input = Input(shape=(self.num_states,),
                        name='input_state')
        l_dense_1 = Dense(num_dense_1,
                          activation='tanh',
                          name='hidden_1')(l_input)
        l_dense_2 = Dense(num_dense_2,
                          activation='tanh',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = Dense(num_dense_3,
                          activation='tanh',
                          name='hidden_3')(l_dense_2)

        l_vs = Dense(1, activation='linear',
                     name='Vs')(l_dense_3)

        model = Model(inputs=[l_input], outputs=[l_vs])
        model.summary()
        return model

    # Critic の計算グラフをコンパイルする
    def _compile_graph(self, model):
        self.state = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.target = tf.placeholder(tf.float32,
                                     shape=(None, 1))

        self.state_value = model(self.state)
        self.loss = tf.squared_difference(
            self.state_value, self.target)
        self.loss = K.mean(self.loss)

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate)
        self.minimize = optimizer.minimize(self.loss)

    # Critic による予測関数を定義
    def predict(self, sess, state):
        return sess.run(self.state_value,
                        {self.state: [state]})

    # Critic の更新関数を定義
    def update(self, sess, state, target):
        feed_dict = {
            self.state: state,
            self.target: target
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
