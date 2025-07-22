import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import shutil
import re



class PolicyGraph():
    """
        Manages the policy computation graph
    """

    def __init__(self, input_states, taken_actions,
                 num_actions, action_min, action_max, scope_name,
                 initial_mean_factor=0.1, clip_action_space=False):
        """
            input_states [batch_size, width, height, depth]:
                Input images to predict actions for
            taken_actions [batch_size, num_actions]:
                Actions taken by the old policy (used for training)
            num_actions (int):
                Number of continuous actions to output
            action_min [num_actions]:
                Minimum possible value for the respective action
            action_max [num_actions]:
                Maximum possible value for the respective action
            scope_name (string):
                Variable scope name for the policy graph
            initial_mean_factor (float):
                Variance scaling factor for the action mean prediction layer
            clip_action_space (bool):
                When True, output actions are clipped to [action_min, action_max] space
        """

        with tf.variable_scope(scope_name):
            # Construct model
            self.conv1 = tf.layers.conv2d(input_states, filters=16, kernel_size=8,
                                          strides=4, activation=tf.nn.leaky_relu, padding="valid", name="conv1")
            self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3,
                                          strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv2")
            self.shared_features = tf.layers.flatten(
                self.conv2, name="flatten")

            # Policy branch π(a_t | s_t; θ)
            self.action_mean = tf.layers.dense(self.shared_features, num_actions,
                                               activation=tf.nn.tanh,
                                               kernel_initializer=tf.initializers.variance_scaling(
                                                   scale=initial_mean_factor),
                                               name="action_mean")
            self.action_mean = action_min + \
                ((self.action_mean + 1) / 2) * (action_max - action_min)
            self.action_logstd = tf.Variable(
                np.full((num_actions), np.log(0.4), dtype=np.float32), name="action_logstd")

            # Value branch V(s_t; θ)
            self.value = tf.layers.dense(
                self.shared_features, 1, activation=None, name="value")

            # Create graph for sampling actions
            self.action_normal = tf.distributions.Normal(
                self.action_mean, tf.exp(self.action_logstd), validate_args=True)
            self.sampled_action = tf.squeeze(
                self.action_normal.sample(1), axis=0)
            if clip_action_space:
                num_envs = tf.shape(self.sampled_action)[0]
                action_min = tf.reshape(tf.tile(tf.convert_to_tensor(
                    action_min, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                action_max = tf.reshape(tf.tile(tf.convert_to_tensor(
                    action_max, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                self.sampled_action = tf.clip_by_value(
                    self.sampled_action, action_min, action_max)

            # Get the log probability of taken actions
            # log π(a_t | s_t; θ)
            self.action_log_prob = tf.reduce_sum(
                self.action_normal.log_prob(taken_actions),
                axis=-1, keepdims=True)

            # Validate values
            self.action_mean = tf.check_numerics(
                self.action_mean, "Invalid value for self.action_mean")
            self.action_logstd = tf.check_numerics(
                self.action_logstd, "Invalid value for self.action_logstd")
            self.value = tf.check_numerics(
                self.value, "Invalid value for self.value")
            self.action_log_prob = tf.check_numerics(
                self.action_log_prob, "Invalid value for self.action_log_prob")


class PPO():
    def __init__(self, input_shape, num_actions, action_min, action_max,
                 epsilon=0.2, value_scale=0.5, entropy_scale=0.01,
                 model_checkpoint=None, model_name="ppo"):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.epsilon = epsilon
        self.value_scale = value_scale
        self.entropy_scale = entropy_scale
        self.model_name = model_name

        # Policy network
        self.policy_model = self.build_policy_model()
        # Old policy network
        self.policy_old_model = self.build_policy_model()
        # Value network
        self.value_model = self.build_value_model()

        self.optimizer = tf.keras.optimizers.Adam()

        # Checkpoint管理
        self.model_dir = f"./models/{self.model_name}"
        self.log_dir = f"./logs/{self.model_name}"
        self.video_dir = f"./videos/{self.model_name}"
        for d in [self.model_dir, self.log_dir, self.video_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)
        self.ckpt = tf.train.Checkpoint(policy=self.policy_model, value=self.value_model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.model_dir, max_to_keep=3)

        if model_checkpoint:
            self.ckpt.restore(model_checkpoint)
            print(f"[INFO] Model checkpoint restored from {model_checkpoint}")

    def build_policy_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        action_mean = layers.Dense(self.num_actions, activation='tanh')(x)
        action_mean = layers.Lambda(lambda x: x * (self.action_max - self.action_min) / 2 + (self.action_max + self.action_min) / 2)(action_mean)
        return keras.Model(inputs=inputs, outputs=action_mean)

    def build_value_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1)(x)
        return keras.Model(inputs=inputs, outputs=value)

    @tf.function
    def train_step(self, states, actions, returns, advantages, old_log_probs, learning_rate):
        with tf.GradientTape() as tape:
            # Forward pass
            action_means = self.policy_model(states, training=True)
            values = tf.squeeze(self.value_model(states, training=True), axis=1)
            # Assume Gaussian policy for continuous control
            std = 1.0  # You may want to learn this as a parameter
            dist = tfp.distributions.Normal(action_means, std)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            # Ratio for PPO
            ratios = tf.exp(log_probs - old_log_probs)
            # PPO policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(
                ratios * advantages,
                tf.clip_by_value(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            ))
            value_loss = tf.reduce_mean(tf.square(values - returns)) * self.value_scale
            entropy_loss = -tf.reduce_mean(entropy) * self.entropy_scale
            total_loss = policy_loss + value_loss + entropy_loss
        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
        self.optimizer.learning_rate.assign(learning_rate)
        self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables + self.value_model.trainable_variables))
        return total_loss

    def save(self):
        model_checkpoint = os.path.join(
            self.model_dir, "step{}.ckpt".format(self.step_idx))
        self.saver.save(self.sess, model_checkpoint)
        print("[INFO] Model checkpoint saved to {}".format(model_checkpoint))

    def train(self, input_states, taken_actions, returns, advantage, learning_rate=1e-4):
        r = self.sess.run([self.summary_merged, self.train_step, self.loss, self.policy_loss, self.value_loss, self.entropy_loss],
                          feed_dict={self.input_states: input_states,
                                     self.taken_actions: taken_actions,
                                     self.returns: returns,
                                     self.advantage: advantage,
                                     self.learning_rate: learning_rate(self.step_idx) if callable(learning_rate) else learning_rate})
        self.train_writer.add_summary(r[0], self.step_idx)
        self.step_idx += 1
        return r[2:]

    def predict(self, input_states, use_old_policy=False, greedy=False):
        policy = self.policy_old if use_old_policy else self.policy
        action = policy.action_mean if greedy else policy.sampled_action
        return self.sess.run([action, policy.value],
                             feed_dict={self.input_states: input_states})

    def write_to_summary(self, name, value):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.train_writer.add_summary(summary, self.step_idx)

    def update_old_policy(self):
        self.sess.run(self.update_op)
