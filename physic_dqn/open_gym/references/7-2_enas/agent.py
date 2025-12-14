import tensorflow as tf


class Agent:

    def __init__(self,
                 num_layers=4,
                 num_operations=6,
                 out_filters=48,
                 lstm_size=32,
                 lstm_num_layers=2,
                 lr_dec_every=100,
                 optim_algo="adam",
                 name="agent",
                 env=None,
                 log_writer=None):

        self.log_writer = log_writer
        self.log_writer.print_and_write("-" * 80)
        self.log_writer.print_and_write("Building Agent")

        self.num_layers = num_layers
        self.num_operations = num_operations
        self.out_filters = out_filters
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = 1.5
        self.temperature = None
        self.lr_dec_every = lr_dec_every
        self.entropy_weight = 0.0001
        self.bl_dec = 0.99
        self.optim_algo = optim_algo
        self.name = name
        self.reward = None
        self.sample_log_prob = None
        self.baseline = None
        self.loss = None
        self.train_step = None
        self.train_op = None
        self.lr = None
        self.grad_norm = None
        self.optimizer = None
        self.env = env

        self._create_params()
        self._build_sampler()

    def _create_params(self):
        initializer = tf.random_uniform_initializer(
            minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name,
                               initializer=initializer):
            with tf.variable_scope("lstm"):
                self.w_lstm = []
                for layer_id in range(
                        self.lstm_num_layers):
                    with tf.variable_scope(
                            "layer_{}".format(layer_id)):
                        w = tf.get_variable(
                            "w", [
                                2 * self.lstm_size,
                                4 * self.lstm_size
                            ])
                        self.w_lstm.append(w)

            self.g_emb = tf.get_variable(
                "g_emb", [1, self.lstm_size])

            with tf.variable_scope("emb"):
                self.w_emb = tf.get_variable(
                    "w", [
                        self.num_operations,
                        self.lstm_size
                    ])
            with tf.variable_scope("softmax"):
                self.w_soft = tf.get_variable(
                    "w", [
                        self.lstm_size,
                        self.num_operations
                    ])

    def _build_sampler(self):
        """Build the sampler ops and the log_prob ops."""

        self.log_writer.print_and_write("-" * 80)
        self.log_writer.print_and_write(
            "Build agent sampler")

        arc_seq = []
        entropys = []
        log_probs = []

        prev_c = [
            tf.zeros([1, self.lstm_size], tf.float32)
            for _ in range(self.lstm_num_layers)
        ]
        prev_h = [
            tf.zeros([1, self.lstm_size], tf.float32)
            for _ in range(self.lstm_num_layers)
        ]
        inputs = self.g_emb

        for layer_id in range(self.num_layers):
            next_c, next_h = self.stack_lstm(
                inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            logit = tf.matmul(next_h[-1], self.w_soft)
            logit = self.tanh_constant * tf.tanh(logit)
            node_operation = tf.multinomial(logit, 1)
            node_operation = tf.to_int32(node_operation)
            node_operation = tf.reshape(
                node_operation, [1])
            arc_seq.append(node_operation)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=node_operation)
            log_probs.append(log_prob)
            entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
            entropys.append(entropy)
            if layer_id == 0:
                continue
            else:
                inputs = tf.nn.embedding_lookup(
                    self.w_emb, node_operation)

        arc_seq = tf.concat(arc_seq, axis=0)
        self.sample_arc = tf.reshape(arc_seq, [-1])

        entropys = tf.stack(entropys)
        self.sample_entropy = tf.reduce_sum(entropys)

        log_probs = tf.stack(log_probs)
        self.sample_log_prob = tf.reduce_sum(log_probs)

    def build_trainer(self):
        self.reward = self.env.valid_shuffle_acc
        self.reward += self.entropy_weight * self.sample_entropy

        self.sample_log_prob = tf.reduce_sum(
            self.sample_log_prob)
        self.baseline = tf.Variable(0.0,
                                    dtype=tf.float32,
                                    trainable=False)
        baseline_update = tf.assign_sub(
            self.baseline, (1 - self.bl_dec) *
            (self.baseline - self.reward))

        with tf.control_dependencies([baseline_update]):
            self.reward = tf.identity(self.reward)

        self.loss = self.sample_log_prob * (
            self.reward - self.baseline)

        self.train_step = tf.Variable(0,
                                      dtype=tf.int32,
                                      trainable=False,
                                      name="train_step")
        tf_variables = [
            var for var in tf.trainable_variables()
            if var.name.startswith(self.name)
        ]
        self.log_writer.print_and_write("-" * 80)
        for var in tf_variables:
            self.log_writer.print_and_write(var)

        (self.train_op, self.lr, self.grad_norm,
         self.optimizer) = self.get_train_ops(
             self.loss,
             tf_variables,
             self.train_step,
             l2_reg=0.0,
             lr_init=0.001,
             lr_dec_start=0,
             lr_dec_every=self.lr_dec_every,
             lr_dec_rate=0.9,
             optim_algo=self.optim_algo)

    def lstm(self, x, prev_c, prev_h, w):
        ifog = tf.matmul(tf.concat([x, prev_h], axis=1),
                         w)
        i, f, o, g = tf.split(ifog, 4, axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        next_c = i * g + f * prev_c
        next_h = o * tf.tanh(next_c)
        return next_c, next_h

    def stack_lstm(self, x, prev_c, prev_h, w):
        next_c, next_h = [], []
        for layer_id, (_c, _h, _w) in enumerate(
                zip(prev_c, prev_h, w)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm(inputs, _c, _h,
                                       _w)
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def get_train_ops(self,
                      loss,
                      tf_variables,
                      train_step,
                      l2_reg=1e-4,
                      lr_init=0.1,
                      lr_dec_start=0,
                      lr_dec_every=10000,
                      lr_dec_rate=0.1,
                      lr_dec_min=None,
                      optim_algo=None):

        if l2_reg > 0:
            l2_losses = []
            for var in tf_variables:
                l2_losses.append(tf.reduce_sum(var**2))
            l2_loss = tf.add_n(l2_losses)
            loss += l2_reg * l2_loss

        grads = tf.gradients(loss, tf_variables)
        grad_norm = tf.global_norm(grads)

        grad_norms = {}
        for v, g in zip(tf_variables, grads):
            if v is None or g is None:
                continue
            if isinstance(g, tf.IndexedSlices):
                grad_norms[v.name] = tf.sqrt(
                    tf.reduce_sum(g.values**2))
            else:
                grad_norms[v.name] = tf.sqrt(
                    tf.reduce_sum(g**2))

        learning_rate = tf.train.exponential_decay(
            lr_init,
            tf.maximum(train_step - lr_dec_start, 0),
            lr_dec_every,
            lr_dec_rate,
            staircase=False)
        if lr_dec_min is not None:
            learning_rate = tf.maximum(learning_rate,
                                       lr_dec_min)

        if optim_algo == "momentum":
            opt = tf.train.MomentumOptimizer(
                learning_rate,
                0.9,
                use_locking=True,
                use_nesterov=True)
        elif optim_algo == "sgd":
            opt = tf.train.GradientDescentOptimizer(
                learning_rate, use_locking=True)
        elif optim_algo == "adam":
            opt = tf.train.AdamOptimizer(
                learning_rate,
                beta1=0.0,
                epsilon=1e-3,
                use_locking=True)
        else:
            raise ValueError(
                "Unknown optim_algo {}".format(
                    optim_algo))

        train_op = opt.apply_gradients(
            zip(grads, tf_variables),
            global_step=train_step)

        return train_op, learning_rate, grad_norm, opt
