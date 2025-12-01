import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class ValueEstimator:

    def __init__(self, dim_state, leaning_rate=1e-3):
        self.dim_state = dim_state
        self.leaning_rate = leaning_rate
        self.build_network()
        self.compile()

    def build_network(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = 5
        nb_dense_2 = int(np.sqrt(nb_dense_1 *
                                 nb_dense_3))

        l_input = Input(shape=(self.dim_state,),
                        name='input_state')
        l_dense_1 = Dense(nb_dense_1,
                          activation='tanh',
                          name='hidden_1')(l_input)
        l_dense_2 = Dense(nb_dense_2,
                          activation='tanh',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = Dense(nb_dense_3,
                          activation='tanh',
                          name='hidden_3')(l_dense_2)
        l_vs = Dense(1, activation='linear',
                     name='Vs')(l_dense_3)

        self.model = Model(inputs=[l_input],
                           outputs=[l_vs])
        self.model.summary()

    def compile(self):
        self.state = tf.placeholder(
            tf.float32, shape=(None, self.dim_state))
        self.target = tf.placeholder(tf.float32,
                                     shape=(None, 1))

        self.state_value = self.model(self.state)
        self.loss = tf.squared_difference(
            self.state_value, self.target)
        self.loss = K.mean(self.loss)

        optimizer = tf.train.AdamOptimizer(
            self.leaning_rate)
        self.minimize = optimizer.minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.state_value,
                        {self.state: [state]})

    def update(self, sess, state, target):
        feed_dict = {
            self.state: state,
            self.target: target
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
