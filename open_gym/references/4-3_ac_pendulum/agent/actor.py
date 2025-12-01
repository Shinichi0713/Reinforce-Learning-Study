import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K


# Actor クラスの定義
class Actor:

    def __init__(self,
                 num_states,
                 actions_list,
                 learning_rate=1e-3):
        self.num_states = num_states
        self.num_actions = len(actions_list)
        self.learning_rate = learning_rate
        self.actions_list = actions_list
        self.model = self._build_network()
        self._compile_graph(self.model)

    # Actor のニューラルネットワーク表現を関数として定義
    def _build_network(self):
        num_dense_1 = self.num_states * 10
        num_dense_3 = self.num_actions * 10
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
        l_prob = Dense(self.num_actions,
                       activation='softmax',
                       name='prob')(l_dense_3)

        model = Model(inputs=[l_input], outputs=[l_prob])
        model.summary()
        return model

    # Actor の計算グラフをコンパイルする
    def _compile_graph(self, model):
        self.state = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.act_onehot = tf.placeholder(
            tf.float32, shape=(None, self.num_actions))
        self.advantage = tf.placeholder(
            tf.float32, shape=(None, 1))

        self.act_prob = model(self.state)
        self.loss = -K.sum(
            K.log(self.act_prob) * self.act_onehot,
            axis=1) * self.advantage
        self.loss = K.mean(self.loss)

        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate)
        self.minimize = optimizer.minimize(self.loss)

    # Actor によるサンプリング関数を定義
    def predict(self, sess, state):
        act_prob = np.array(
            sess.run([self.act_prob],
                     {self.state: [state]}))
        action = [
            np.random.choice(self.actions_list,
                             p=prob[0])
            for prob in act_prob
        ]
        return action[0]

    # Actor の更新関数を定義
    def update(self, sess, state, act_onehot, advantage):
        feed_dict = {
            self.state: state,
            self.act_onehot: act_onehot,
            self.advantage: advantage
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
