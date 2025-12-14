import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class PolicyEstimator:

    def __init__(self,
                 dim_state,
                 dim_action,
                 leaning_rate=1e-3):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.leaning_rate = leaning_rate
        self.build_network()
        self.compile()

    def build_network(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.dim_action * 10
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
        l_mu = Dense(self.dim_action,
                     activation='tanh',
                     name='mu')(l_dense_3)
        l_log_var = Dense(self.dim_action,
                          activation='tanh',
                          name='log_var')(l_dense_3)

        self.model = Model(inputs=[l_input],
                           outputs=[l_mu, l_log_var])
        self.model.summary()

    def compile(self):
        self.state = tf.placeholder(
            tf.float32, shape=(None, self.dim_state))
        self.action = tf.placeholder(
            tf.float32, shape=(None, self.dim_action))
        self.advantage = tf.placeholder(tf.float32,
                                        shape=(None, 1))

        self.mu, self.log_var = self.model(self.state)

        self.action_logprobs = self.logprob()
        self.loss = -self.action_logprobs * self.advantage
        self.loss = K.mean(self.loss)

        optimizer = tf.train.RMSPropOptimizer(
            self.leaning_rate)
        self.minimize = optimizer.minimize(self.loss)

    def logprob(self):
        action_logprobs = -0.5 * self.log_var
        action_logprobs += -0.5 \
            * K.square(self.action - self.mu) \
            / K.exp(self.log_var)
        return action_logprobs

    def predict(self, sess, state):
        mu, log_var = sess.run([self.mu, self.log_var],
                               {self.state: [state]})
        mu, log_var = mu[0], log_var[0]
        var = np.exp(log_var)
        action = np.random.normal(loc=mu,
                                  scale=np.sqrt(var))
        return action

    def update(self, sess, state, action, advantage):
        feed_dict = {
            self.state: state,
            self.action: action,
            self.advantage: advantage
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)
        return loss
