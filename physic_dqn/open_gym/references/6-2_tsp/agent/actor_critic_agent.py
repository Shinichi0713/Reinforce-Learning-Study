import os

import tensorflow as tf

from agent.models import Encoder, ActorDecoder, CriticDecoder
from agent.losses import rms_loss, policy_gradient_loss
from agent.losses import tour_distance


class ActorCriticAgent(object):

    def __init__(self, n_neurons=128, batch_size=4, seq_length=10, coord_dim=2,
                 critic_learning_rate=1.0E-03, actor_learning_rate=1.0E-03):

        # パラメタ設定
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.coord_dim = coord_dim
        self.val_lr = critic_learning_rate
        self.pol_lr = actor_learning_rate

        # 入力データの placeholder の定義
        data_dim = (self.seq_length, self.coord_dim)
        input = tf.placeholder(shape=(None, *data_dim),
                               dtype=tf.float32,
                               name='input_data')
        self.p_holders = input

        # 共通Encoderネットワークの構成
        self.encoder = Encoder()
        enc_outputs, enc_state = \
            self.encoder.build_model(input)

        # Encoder出力に基づいて方策関数(Actor)のネットワークを構成
        self.actor_decoder = ActorDecoder()
        log_prob, tour = \
            self.actor_decoder.build_model(enc_outputs,
                                           enc_state)

        # Encoder出力に基づいて価値関数(Critic)のネットワークを構成
        self.critic_decoder = CriticDecoder()
        state_value = \
            self.critic_decoder.build_model(enc_outputs,
                                            enc_state)

        # 巡回路ツアーの距離（報酬）の構成
        tour_dist = tour_distance(input, tour)
        self.rewards = [-1.0 * tour_dist, tour_dist]

        # lossの構成
        self.model_prds = [log_prob, tour, state_value]
        loss, vloss, ploss = self._build_loss(self.model_prds, self.rewards)
        self.losses = [loss, vloss, ploss]

        # optimizerの構成
        v_optim, v_grad = self._build_value_optimizer(vloss)
        p_optim, p_grad = self._build_policy_optimizer(ploss)
        self.opts = [v_optim, p_optim]

        # graph variableのsaverの定義
        self.saver = self._build_graph_saver()

    # lossの構成
    def _build_loss(self, model_prds, rewards):

        log_prob, tour, baseline = model_prds
        reward, tour_dist = rewards

        # advantage for PG
        _advantage = -1.0 * tf.stop_gradient(tour_dist - baseline)

        # actor/critic losses
        vloss = rms_loss(tour_dist, baseline)
        ploss = policy_gradient_loss(_advantage, log_prob)

        # loss weight
        loss_wt = [1.0, 1.0]
        loss = loss_wt[0] * vloss + loss_wt[1] * ploss

        return loss, vloss, ploss

    # criticのoptimizerの構成
    def _build_value_optimizer(self, val_loss):

        val_optimizer = tf.train.RMSPropOptimizer(self.val_lr)

        val_optim = val_optimizer.minimize(val_loss)
        val_grad = val_optimizer.compute_gradients(val_loss)

        return val_optim, val_grad

    # actorのoptimizerの構成
    def _build_policy_optimizer(self, pol_loss):

        pol_optimizer = tf.train.RMSPropOptimizer(self.pol_lr)

        pol_optim = pol_optimizer.minimize(pol_loss)
        pol_grad = pol_optimizer.compute_gradients(pol_loss)

        return pol_optim, pol_grad

    # 変数saverの構成
    def _build_graph_saver(self):
        variables_to_save = tf.global_variables()
        return tf.train.Saver(var_list=variables_to_save)

    def save_graph(self, sess, log_dir, args):
        fname = 'model.{0:06d}-{1:3.3f}-{2:3.5f}.ckpt'.format(*args)
        self.saver.save(sess, os.path.join(log_dir, fname))

    def restore_graph(self, sess, model_path):
        self.saver.restore(sess, model_path)

    # モデル更新（agent.predict、env.step、agent.updateの一括実行に相当）
    def update_model(self, sess, state):

        input_data = self.p_holders
        v_optim, p_optim = self.opts

        feed_dict = {input_data: state}
        # criticの更新
        tensors = [v_optim, self.losses, self.rewards, self.model_prds]
        _, losses, rewards, mode_prds = sess.run(tensors, feed_dict)
        # actorの更新
        tensors = [p_optim, self.losses, self.rewards, self.model_prds]
        _, losses, rewards, mode_prds = sess.run(tensors, feed_dict)

        return losses, rewards, mode_prds

    # loss推定（agent.predict、env.stepの一括実行に相当）
    def predict_loss(self, sess, state):

        input_data = self.p_holders

        feed_dict = {input_data: state}
        tensors = [self.losses, self.rewards, self.model_prds]
        losses, rewards, model_prds = sess.run(tensors, feed_dict)

        return losses, rewards, model_prds
