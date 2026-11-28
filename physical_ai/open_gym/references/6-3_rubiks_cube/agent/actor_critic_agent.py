import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from agent.models import build_model
from agent.losses import huber_loss, policy_gradient_loss


# エージェントクラスの定義
class ActorCriticAgent(object):

    def __init__(self,
                 state_shape,
                 action_list,
                 gamma=0.99,
                 critic_learning_rate=1.0E-05,
                 actor_learning_rate=1.0E-04):
        # setup variables
        self.input_shape = state_shape

        self.action_list = action_list
        self.num_value = 1
        self.num_action = len(action_list)

        self.gamma = gamma
        self.val_lr = critic_learning_rate
        self.pol_lr = actor_learning_rate

        # placeholders
        input, val_obs, td_err, act_obs = self._build_placeholder(
        )
        self.p_holders = [
            input, val_obs, td_err, act_obs
        ]

        # model predictions
        value, act_prob = self._build_agent_network(
            input)
        self.model_prds = [value, act_prob]

        # losses
        loss, vloss, ploss = self._build_loss(
            val_obs, td_err, act_obs, value, act_prob)
        self.losses = [loss, vloss, ploss]

        # optimizers
        v_optim, v_grad = self._build_value_optimizer(
            vloss)
        p_optim, p_grad = self._build_policy_optimizer(
            ploss)
        self.opts = [v_optim, p_optim]

        # graph variable saver
        self.saver = self._build_graph_saver()

    # build placeholders, and return associated tensors
    def _build_placeholder(self):

        # placeholders
        input = tf.placeholder(shape=(None,
                               *self.input_shape),
                               dtype=tf.float32,
                               name='input')
        val_obs = tf.placeholder(shape=(None,
                                        self.num_value),
                                 dtype=tf.float32,
                                 name='val_obs')
        td_err = tf.placeholder(shape=(None,
                                       self.num_value),
                                dtype=tf.float32,
                                name='td_err')
        act_obs = tf.placeholder(shape=(None,
                                        self.num_action),
                                 dtype=tf.float32,
                                 name='act_obs')

        return input, val_obs, td_err, act_obs

    # build actor/critic networks, and return associated tensors
    def _build_agent_network(self, input_data):

        critic_model, actor_model = build_model(
            self.input_shape, self.num_value,
            self.num_action)
        value = critic_model(input_data)
        action_prob = actor_model(input_data)

        return value, action_prob

    # build loss, and return associated tensors
    def _build_loss(self, val_obs, td_err, act_obs,
                    val_prd, act_prd):

        # actor/critic losses
        vloss = huber_loss(val_obs, val_prd)
        ploss = policy_gradient_loss(act_obs, td_err,
                                     act_prd)

        # loss weight
        loss_wt = [1.0, 1.0]
        loss = loss_wt[0] * vloss + loss_wt[1] * ploss

        return loss, vloss, ploss

    # build critic optimizer, and return associated tensors
    def _build_value_optimizer(self, val_loss):

        val_optimizer = tf.train.RMSPropOptimizer(
            self.val_lr)

        val_optim = val_optimizer.minimize(val_loss)
        val_grad = val_optimizer.compute_gradients(
            val_loss)

        return val_optim, val_grad

    # build actor optimizer, and return associated tensors
    def _build_policy_optimizer(self, pol_loss):

        pol_optimizer = tf.train.RMSPropOptimizer(
            self.pol_lr)

        pol_optim = pol_optimizer.minimize(pol_loss)
        pol_grad = pol_optimizer.compute_gradients(
            pol_loss)

        return pol_optim, pol_grad

    # build variable saver for graph
    def _build_graph_saver(self):
        variables_to_save = tf.global_variables()
        return tf.train.Saver(var_list=variables_to_save)

    def save_graph(self, sess, log_dir, args):
        fname = 'model.{0:06d}-{1:3.3f}-{2:3.5f}.ckpt'.format(
            *args)
        self.saver.save(sess,
                        os.path.join(log_dir, fname))

    def restore_graph(self, sess, model_path):
        self.saver.restore(sess, model_path)

    # モデル更新
    def update_model(self, sess, state, action, reward,
                     next_state, done):

        # one-hot arrayへの変換
        action_idx = [
            self.action_list.index(act) for act in action
        ]
        action_idx = to_categorical(action_idx,
                                    self.num_action)

        # 状態価値とTD誤差の算出
        # TD(0)アルゴリズムの場合
        if 0:
            next_st_val = self.predict_value(
                sess, next_state)
            target_val = np.where(done, reward,
                                  reward +
                                  self.gamma * next_st_val)
        # TD(λ)アルゴリズムの場合
        if 1:
            # terminal state-valueの加算（if not done）
            _v_s = ([[0.0]] if done[-1][0]
                    else self.predict_value(
                        sess, [next_state[-1]]))
            terminal_val = self.gamma**(
                len(reward)) * _v_s[0][0]

            # 報酬値の積算によるGtの算出
            target_val = []
            for i_step in range(len(reward)):
                rwd_seq = [
                    self.gamma**i * i_rwd[0]
                    for i, i_rwd in enumerate(
                        reward[i_step:])]
                rwd_seq += [terminal_val]

                g_t = np.sum(rwd_seq)
                target_val.append([g_t])

        st_val = self.predict_value(sess, state)
        td_error = target_val - st_val

        input, val_obs, td_err, act_obs = self.p_holders
        v_optim, p_optim = self.opts

        # feed_dictの定義
        feed_dict = {
            input: state,
            val_obs: target_val,
            td_err: td_error,
            act_obs: action_idx
        }
        # 価値ネットワークの更新
        _, losses = sess.run(
            [v_optim, self.losses], feed_dict)
        # 方策ネットワークの更新
        _, losses = sess.run(
            [p_optim, self.losses], feed_dict)

        return losses

    # critic prediction
    def predict_value(self, sess, state):

        value, act_prob = self.model_prds
        input, val_obs, td_err, act_obs = self.p_holders

        feed_dict = {input: state}
        state_value = sess.run(value, feed_dict)

        return state_value

    # actor prediction
    def predict_policy(self, sess, state):

        value, act_prob = self.model_prds
        input, val_obs, td_err, act_obs = self.p_holders

        feed_dict = {input: state}
        action_prob = sess.run(act_prob, feed_dict)

        return action_prob

    # get rollout result for 1-D input
    def roll_out(self, sess, env, steps, state):

        extra_reward = 0.0
        _state = state.copy()
        done = [False]

        # NOTE: This can be multiprocessed to reduce variance
        for i_step in range(steps):
            # obtain action prediction
            action = self.get_action(sess, _state)
            # obtain reward for the action
            next_state, reward, done, _ = env.step(
                action)
            # calc weighted reward and accumulate
            extra_reward += self.gamma**(
                i_step) * reward[0]
            # update state
            _state = next_state
            # break if reached to done
            if done[0]:
                break

        # add terminal state value if its not done
        if not done[0]:
            _v_s = self.predict_value(sess, [_state])
            extra_reward += self.gamma**(
                steps) * _v_s[0][0]

        # additional gamma factor
        extra_reward *= self.gamma

        # set back cube state
        env.set_state(state)

        return extra_reward

    # get action for 1-D input
    def get_action(self, sess, state):

        value, act_prob = self.model_prds
        input, val_obs, td_err, act_obs = self.p_holders

        feed_dict = {input: [state]}
        action_prob = sess.run(act_prob, feed_dict)

        action = np.random.choice(self.action_list,
                                  p=action_prob[0])

        return action

    # get action for 1-D input
    def get_greedy_action(self, sess, state):

        value, act_prob = self.model_prds
        input, val_obs, td_err, act_obs = self.p_holders

        feed_dict = {input: [state]}
        action_prob = sess.run(act_prob, feed_dict)

        action = self.action_list[np.argmax(action_prob)]

        return action

    # loss推定
    def predict_loss(self, sess, state, action, reward,
                     next_state, done):

        # one-hot arrayへの変換
        action_idx = [
            self.action_list.index(act) for act in action
        ]
        action_idx = to_categorical(action_idx,
                                    self.num_action)

        # 状態価値とTD誤差の算出
        # TD(0)アルゴリズムの場合
        if 0:
            next_st_val = self.predict_value(
                sess, next_state)
            target_val = np.where(
                done, reward,
                reward + self.gamma * next_st_val)
        # TD(λ)アルゴリズムの場合
        if 1:
            # terminal state-valueの加算（if not done）
            _v_s = ([[0.0]] if done[-1][0]
                    else self.predict_value(
                        sess, [next_state[-1]]))
            terminal_val = self.gamma**(
                len(reward)) * _v_s[0][0]

            # 報酬値の積算によるGtの算出
            target_val = []
            for i_step in range(len(reward)):
                rwd_seq = [
                    self.gamma**i * i_rwd[0] for i, i_rwd
                    in enumerate(reward[i_step:])
                ]
                rwd_seq += [terminal_val]

                g_t = np.sum(rwd_seq)
                target_val.append([g_t])

        st_val = self.predict_value(sess, state)
        td_error = target_val - st_val

        input, val_obs, td_err, act_obs = self.p_holders

        # feed_dictの定義
        feed_dict = {
            input: state,
            val_obs: target_val,
            td_err: td_error,
            act_obs: action_idx
        }
        # lossの算出
        losses = sess.run(self.losses, feed_dict)

        return losses
