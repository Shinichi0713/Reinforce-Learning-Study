import math
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from PIL import Image


class Environment:

    def __init__(self,
                 fixed_arc=None,
                 num_layers=2,
                 num_operations=6,
                 out_filters=24,
                 batch_size=32,
                 name="environment",
                 loss_op="CE",
                 accuracy_op="IOU",
                 kernel_size=(3, 5),
                 dilate_rate=(1, 1),
                 eval_batch_size=100,
                 image_dir=None,
                 label_dir=None,
                 log_writer=None):

        self.lr_dec_every = 50000
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.loss_op = loss_op
        self.accuracy_op = accuracy_op
        self.kernel_size = kernel_size
        self.dilate_rate = dilate_rate
        self.num_operations = num_operations
        self.fixed_arc = fixed_arc
        self.pool_distance = self.num_layers // 4
        self.pool_layers = [
            self.pool_distance - 1,
            2 * self.pool_distance - 1,
            3 * self.pool_distance - 1
        ]
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.name = name
        self.global_step = None
        self.valid_acc = None
        self.test_acc = None
        self.img_path = image_dir
        self.lbl_path = label_dir
        self.log_writer = log_writer

    def _model(self, images, is_training, reuse=False):
        """
        Builds the environment model
        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope(self.name, reuse=reuse):
            layers = []

            out_filters = self.out_filters
            with tf.variable_scope("stem_conv"):
                w = self.create_weight(
                    "w", [3, 3, 3, out_filters])
                x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME")
                x = self.batch_norm(x, is_training)
                layers.append(x)

            start_idx = 0

            # Build encoder
            for layer_id in range(self.num_layers):
                with tf.variable_scope(
                        "encoder_layer_{0}".format(layer_id)):
                    x = self._enas_layer(
                        layers, start_idx, out_filters,
                        is_training)
                    if layer_id in self.pool_layers:
                        x = tf.layers.max_pooling2d(
                            inputs=x,
                            pool_size=[2, 2],
                            strides=[2, 2],
                            padding="same")
                    layers.append(x)
                start_idx += 1
            start_idx -= 1

            # Build decoder with long skip
            for layer_id in reversed(
                    range(self.num_layers)):
                with tf.variable_scope(
                        "decoder_layer_{0}".format(
                            2 * self.num_layers -
                            layer_id - 1)):
                    x = self._enas_layer(
                        layers, start_idx, out_filters,
                        is_training)
                    if layer_id in self.pool_layers:
                        with tf.variable_scope(
                                "concat_layer"):
                            x = tf.image.resize_nearest_neighbor(
                                x,
                                size=[
                                    x.get_shape()[1] * 2,
                                    x.get_shape()[2] * 2
                                ],
                                align_corners=True)
                            w = self.create_weight(
                                "w", [
                                    1, 1,
                                    2 * out_filters,
                                    out_filters
                                ])
                            x = tf.concat([
                                x, layers[layer_id - 1]
                            ],
                                          axis=3)
                            x = tf.nn.conv2d(
                                x, w, [1, 1, 1, 1],
                                "SAME")
                    layers.append(x)
                start_idx -= 1

            with tf.variable_scope("end_conv"):
                w = self.create_weight(
                    "w", [1, 1, out_filters, 22])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1],
                                 "SAME")

        return x

    def _enas_layer(self, prev_layers, start_idx,
                    out_filters, is_training):
        """
        Args:
          layer_id: current layer
          prev_layers: cache of previous layers. for skip connections
          start_idx: where to start looking at. technically, we can infer this
            from layer_id, but why bother...
          is_training: for batch_norm
        """
        inputs = prev_layers[-1]
        inp_h = inputs.get_shape()[1].value
        inp_w = inputs.get_shape()[2].value

        count = self.sample_arc[start_idx]
        branches = {}
        with tf.variable_scope("operation_0"):
            y = self._conv_operation(
                inputs,
                int(self.kernel_size[0]),
                is_training,
                out_filters,
                out_filters,
                start_idx=0,
                conv_type="astrous",
                rate=int(self.dilate_rate[0]))
            branches[tf.equal(count, 0)] = lambda: y
        with tf.variable_scope("operation_1"):
            y = self._conv_operation(
                inputs,
                3,
                is_training,
                out_filters,
                out_filters,
                start_idx=0,
                conv_type="separable")
            branches[tf.equal(count, 1)] = lambda: y
        with tf.variable_scope("operation_2"):
            y = self._conv_operation(
                inputs,
                int(self.kernel_size[1]),
                is_training,
                out_filters,
                out_filters,
                start_idx=0,
                conv_type="astrous",
                rate=int(self.dilate_rate[1]))
            branches[tf.equal(count, 2)] = lambda: y
        with tf.variable_scope("operation_3"):
            y = self._conv_operation(
                inputs,
                5,
                is_training,
                out_filters,
                out_filters,
                start_idx=0,
                conv_type="separable")
            branches[tf.equal(count, 3)] = lambda: y
        if self.num_operations >= 5:
            with tf.variable_scope("operation_4"):
                y = self._pool_operation(inputs,
                                         is_training,
                                         out_filters,
                                         "avg",
                                         start_idx=0)
            branches[tf.equal(count, 4)] = lambda: y
        if self.num_operations >= 6:
            with tf.variable_scope("operation_5"):
                y = self._pool_operation(inputs,
                                         is_training,
                                         out_filters,
                                         "max",
                                         start_idx=0)
            branches[tf.equal(count, 5)] = lambda: y

        out_shape = [
            self.batch_size, inp_h, inp_w, out_filters
        ]

        out = tf.case(
            branches,
            default=lambda: tf.constant(
                0, tf.float32, shape=out_shape),
            exclusive=True)

        return out

    def _conv_operation(self,
                        inputs,
                        filter_size,
                        is_training,
                        count,
                        out_filters,
                        ch_mul=1,
                        start_idx=None,
                        conv_type="conv",
                        rate=1):
        """
        Args:
          start_idx: where to start taking the output channels.
                     if None, assume fixed_arc mode
          count: how many output_channels to take.
        """
        if start_idx is None:
            assert self.fixed_arc is not None, "Error!"

        inp_c = inputs.get_shape()[3].value

        with tf.variable_scope("inp_conv_1"):
            w = self.create_weight(
                "w", [1, 1, inp_c, out_filters])
            x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1],
                             "SAME")
            x = self.batch_norm(x, is_training)
            x = tf.nn.relu(x)

        with tf.variable_scope(
                "out_conv_{}".format(filter_size)):
            if start_idx is None:
                if conv_type == "separable":
                    w_depth = self.create_weight(
                        "w_depth", [
                            self.filter_size,
                            self.filter_size,
                            out_filters, ch_mul
                        ])
                    w_point = self.create_weight(
                        "w_point", [
                            1, 1, out_filters * ch_mul,
                            count
                        ])
                    x = tf.nn.separable_conv2d(
                        x,
                        w_depth,
                        w_point,
                        strides=[1, 1, 1, 1],
                        padding="SAME")
                    x = self.batch_norm(x, is_training)
                elif conv_type == "astrous":
                    w = self.create_weight(
                        "w", [
                            filter_size, filter_size,
                            inp_c, count
                        ])
                    x = tf.nn.atrous_conv2d(
                        x, w, rate, "SAME")
                    x = self.batch_norm(x, is_training)
                else:
                    w = self.create_weight(
                        "w", [
                            filter_size, filter_size,
                            inp_c, count
                        ])
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1],
                                     "SAME")
                    x = self.batch_norm(x, is_training)
            else:
                if conv_type == "separable":
                    w_depth = self.create_weight(
                        "w_depth", [
                            filter_size, filter_size,
                            out_filters, ch_mul
                        ])
                    w_point = self.create_weight(
                        "w_point", [
                            out_filters,
                            out_filters * ch_mul
                        ])
                    w_point = w_point[
                        start_idx:start_idx + count, :]
                    w_point = tf.transpose(
                        w_point, [1, 0])
                    w_point = tf.reshape(
                        w_point, [
                            1, 1, out_filters * ch_mul,
                            count
                        ])

                    x = tf.nn.separable_conv2d(
                        x,
                        w_depth,
                        w_point,
                        strides=[1, 1, 1, 1],
                        padding="SAME")
                    mask = tf.range(0,
                                    out_filters,
                                    dtype=tf.int32)
                    mask = tf.logical_and(
                        start_idx <= mask,
                        mask < start_idx + count)
                    x = self.batch_norm_with_mask(
                        x, is_training, mask,
                        out_filters)
                elif conv_type == "astrous":
                    w = self.create_weight(
                        "w", [
                            filter_size, filter_size,
                            out_filters, out_filters
                        ])
                    w = w[:, :, :,
                          start_idx:start_idx + count]
                    x = tf.nn.atrous_conv2d(
                        x, w, rate, "SAME")
                    mask = tf.range(0,
                                    out_filters,
                                    dtype=tf.int32)
                    mask = tf.logical_and(
                        start_idx <= mask,
                        mask < start_idx + count)
                    x = self.batch_norm_with_mask(
                        x, is_training, mask,
                        out_filters)
                else:
                    w = self.create_weight(
                        "w", [
                            filter_size, filter_size,
                            out_filters, out_filters
                        ])
                    w = w[:, :, :,
                          start_idx:start_idx + count]
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1],
                                     "SAME")
                    mask = tf.range(0,
                                    out_filters,
                                    dtype=tf.int32)
                    mask = tf.logical_and(
                        start_idx <= mask,
                        mask < start_idx + count)
                    x = self.batch_norm_with_mask(
                        x, is_training, mask,
                        out_filters)
            x = tf.nn.relu(x)

        return x

    def _pool_operation(self,
                        inputs,
                        is_training,
                        count,
                        avg_or_max,
                        start_idx=None):
        """
        Args:
          start_idx: where to start taking the output channels.
                     if None, assume fixed_arc mode
          count: how many output_channels to take.
        """
        if start_idx is None:
            assert self.fixed_arc is not None, "you screwed up!"

        inp_c = inputs.get_shape()[3].value

        with tf.variable_scope("conv_1"):
            w = self.create_weight(
                "w", [1, 1, inp_c, self.out_filters])
            x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1],
                             "SAME")
            x = self.batch_norm(x, is_training)
            x = tf.nn.relu(x)

        with tf.variable_scope("pool"):
            if avg_or_max == "avg":
                x = tf.layers.average_pooling2d(
                    x, [3, 3], [1, 1], "SAME")
            elif avg_or_max == "max":
                x = tf.layers.max_pooling2d(
                    x, [3, 3], [1, 1], "SAME")
            else:
                raise ValueError(
                    "Unknown pool {}".format(avg_or_max))

            if start_idx is not None:
                x = x[:, :, :,
                      start_idx:start_idx + count]

        return x

    def _calculate_accuracy(self, probs, y):
        if self.accuracy_op == "ACC":
            correct_prediction = tf.equal(
                tf.argmax(probs, axis=3),
                tf.argmax(y, axis=3))
            return tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
        else:
            one_hot = tf.one_hot(
                tf.nn.top_k(self.train_preds).indices,
                tf.shape(self.train_preds)[3])
            one_hot = tf.reshape(one_hot,
                                 shape=tf.shape(
                                     self.train_preds))
            temp_y = tf.cast(y, tf.bool)
            if self.accuracy_op == "IOU1":
                TP = tf.reduce_sum(tf.cast(
                    tf.logical_and(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                   axis=[0, 1, 2])
                FPFN = tf.reduce_sum(tf.cast(
                    tf.logical_xor(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                     axis=[0, 1, 2])
                IOU = TP / (TP + FPFN)
                IOU = tf.where(tf.is_nan(IOU),
                               tf.zeros_like(IOU), IOU)
                IOU = tf.reduce_mean(IOU)
            elif self.accuracy_op == "IOU2":
                TP = tf.reduce_sum(tf.cast(
                    tf.logical_and(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                   axis=[1, 2])
                FPFN = tf.reduce_sum(tf.cast(
                    tf.logical_xor(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                     axis=[1, 2])
                TP1 = TP[:, 0]
                FPFN1 = FPFN[:, 0]
                IOU1 = TP1 / (TP1 + FPFN1)
                IOU1 = tf.where(tf.is_nan(IOU1),
                                tf.zeros_like(IOU1),
                                IOU1)
                TP2 = tf.reduce_sum(TP, axis=1) - TP1
                FPFN2 = tf.reduce_sum(FPFN,
                                      axis=1) - FPFN1
                IOU2 = TP2 / (TP2 + FPFN2)
                IOU2 = tf.where(tf.is_nan(IOU2),
                                tf.zeros_like(IOU2),
                                IOU2)
                IOU = tf.reduce_mean(IOU1 + IOU2)
            elif self.accuracy_op == "IOU3":
                TP = tf.reduce_sum(tf.cast(
                    tf.logical_and(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                   axis=[0, 1, 2])
                FPFN = tf.reduce_sum(tf.cast(
                    tf.logical_xor(
                        temp_y,
                        tf.cast(one_hot, tf.bool)),
                    tf.float32),
                                     axis=[0, 1, 2])
                TP1 = TP[0]
                FPFN1 = FPFN[0]
                IOU1 = TP1 / (TP1 + FPFN1)
                IOU1 = tf.where(tf.is_nan(IOU1),
                                tf.zeros_like(IOU1),
                                IOU1)
                TP2 = tf.reduce_sum(TP) - TP1
                FPFN2 = tf.reduce_sum(FPFN) - FPFN1
                IOU2 = TP2 / (TP2 + FPFN2)
                IOU2 = tf.where(tf.is_nan(IOU2),
                                tf.zeros_like(IOU2),
                                IOU2)
                IOU = tf.reduce_mean(IOU1 + IOU2)
            return IOU

    def _build_train(self):
        self.log_writer.print_and_write("-" * 80)
        self.log_writer.print_and_write(
            "Build train graph")
        logits = self._model(self.x_train,
                             is_training=True)
        probs = tf.nn.softmax(logits, axis=-1)
        self.train_preds = probs

        # Calculate loss with either Mean Square Error (MSE)
        # or Cross Entropy (CE)
        if self.loss_op == "MSE":
            truth = tf.cast(self.y_train, tf.float32)
            mse = tf.metrics.mean_squared_error(
                labels=truth, predictions=probs)
            self.loss = tf.reduce_mean(mse)
        elif self.loss_op == "CE":
            neg_log = \
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits,
                    labels=self.y_train)
            self.loss = tf.reduce_mean(neg_log)
        else:
            raise ValueError(
                "Loss op must be either 'MSE' or 'CE'")

        # Calculate accuracy
        self.train_acc = self._calculate_accuracy(
            probs, self.y_train)

        tf_variables = [
            var for var in tf.trainable_variables()
            if var.name.startswith(self.name)
        ]
        self.num_vars = self.count_model_params(
            tf_variables)
        self.log_writer.print_and_write(
            "Model has {} params".format(self.num_vars))

        self.global_step = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="global_step")
        (self.train_op, self.lr, self.grad_norm,
         self.optimizer) = self.get_train_ops(
             self.loss,
             tf_variables,
             self.global_step,
             clip_mode="norm",
             grad_bound=5.0,
             l2_reg=0.00025,
             lr_init=0.05,
             lr_dec_start=0,
             lr_dec_every=self.lr_dec_every,
             lr_dec_rate=0.1,
             lr_cosine=True,
             lr_max=0.05,
             lr_min=0.0005,
             lr_T_0=10,
             lr_T_mul=2,
             num_train_batches=self.num_train_batches,
             optim_algo="momentum")

    def _build_valid(self):
        if self.x_valid is not None:
            self.log_writer.print_and_write("-" * 80)
            self.log_writer.print_and_write(
                "Build valid graph")
            logits = self._model(self.x_valid,
                                 False,
                                 reuse=True)
            probs = tf.nn.softmax(logits, axis=-1)
            self.valid_preds = probs

            probs = tf.nn.softmax(logits, axis=-1)
            self.valid_preds = probs

            # Calculate accuracy
            self.valid_acc = self._calculate_accuracy(
                probs, self.y_valid)

    def _build_test(self):
        self.log_writer.print_and_write("-" * 80)
        self.log_writer.print_and_write(
            "Build test graph")
        logits = self._model(self.x_test,
                             False,
                             reuse=True)
        probs = tf.nn.softmax(logits, axis=-1)
        self.test_preds = probs

        probs = tf.nn.softmax(logits, axis=-1)
        self.test_preds = probs

        # Calculate accuracy
        self.test_acc = self._calculate_accuracy(
            probs, self.y_test)

    def _build_valid_rl(self):
        self.log_writer.print_and_write("-" * 80)
        self.log_writer.print_and_write(
            "Build valid_rl graph")
        logits = self._model(self.x_valid,
                             False,
                             reuse=True)
        probs = tf.nn.softmax(logits, axis=-1)

        # Calculate accuracy
        self.valid_shuffle_acc = self._calculate_accuracy(
            probs, self.y_valid)

    def connect_agent(self, agent_model):
        if self.fixed_arc is None:
            self.sample_arc = agent_model.sample_arc
        else:
            fixed_arc = np.array([
                int(x)
                for x in self.fixed_arc.split(" ")
                if x
            ])
            self.sample_arc = fixed_arc

        self._build_data()
        self._build_train()
        self._build_valid()
        self._build_test()
        self._build_valid_rl()

    def eval_once(self,
                  sess,
                  eval_set,
                  feed_dict=None,
                  verbose=False):
        """Expects self.acc and self.global_step to be defined.

        Args:
          sess: tf.Session() or one of its wrap arounds.
          feed_dict: can be used to give more information to sess.run().
          eval_set: "valid" or "test"
        """

        assert self.global_step is not None
        global_step = sess.run(self.global_step)
        self.log_writer.print_and_write(
            "Eval at {}".format(global_step))

        if eval_set == "valid":
            assert self.x_valid is not None
            assert self.valid_acc is not None
            num_batches = self.num_valid_batches
            acc_op = self.valid_acc
        elif eval_set == "test":
            assert self.test_acc is not None
            num_batches = self.num_test_batches
            acc_op = self.test_acc
        else:
            raise NotImplementedError(
                "Unknown eval_set '{}'".format(eval_set))

        total_acc = 0
        total_exp = 0
        for batch_id in range(num_batches):
            acc = sess.run(acc_op, feed_dict=feed_dict)
            total_acc += acc
            total_exp += 1
            if verbose:
                sys.stdout.write(
                    "\r{:<5d}/{:>5d}".format(
                        total_acc, total_exp))
        if verbose:
            self.log_writer.print_and_write("")
        self.log_writer.print_and_write(
            "{}_accuracy: {:<6.4f}".format(
                eval_set,
                float(total_acc) / total_exp))

    def _build_data(self):
        with tf.device("/cpu:0"):
            # Only use names from lbl_path since not all images are labeled
            lbl_filename_list = os.listdir(self.lbl_path)
            lbl_filename_list.sort()
            img_filename_list = [
                lbl_filename.replace(".png", ".jpg")
                for lbl_filename in lbl_filename_list
            ]

            # Parameters
            height = 128
            width = 128
            category = 22

            # 70% train, 20% validation, 10% test
            num_images = len(img_filename_list)
            num_train = int(num_images * 0.7)
            num_valid = int(num_images * 0.2)
            num_test = num_images - num_train - num_valid
            train_img_filename_list = img_filename_list[
                0:num_train]
            valid_img_filename_list = \
                img_filename_list[num_train:num_train + num_valid]
            test_img_filename_list = \
                img_filename_list[num_images - num_test:num_images]
            train_lbl_filename_list = lbl_filename_list[
                0:num_train]
            valid_lbl_filename_list = \
                lbl_filename_list[num_train:num_train + num_valid]
            test_lbl_filename_list = \
                lbl_filename_list[num_images - num_test:num_images]

            # Create TensorFlow Dataset objects
            train_data = tf.data.Dataset.from_tensor_slices(
                (train_img_filename_list,
                 train_lbl_filename_list))
            valid_data = tf.data.Dataset.from_tensor_slices(
                (valid_img_filename_list,
                 valid_lbl_filename_list))
            test_data = tf.data.Dataset.from_tensor_slices(
                (test_img_filename_list,
                 test_lbl_filename_list))

            # Load labels using PIL, and convert them to one-hot vector
            def read_labels(lbl_filename):
                train_lbl = Image.open(
                    self.lbl_path +
                    lbl_filename.decode("utf-8"))
                train_lbl = train_lbl.resize(
                    (height, width))
                train_lbl = np.asarray(train_lbl)

                # Change indices which correspond to "void" from 255
                train_lbl = np.where(train_lbl == 255,
                                     21, train_lbl)

                # Convert to one hot encoding
                identity = np.identity(category,
                                       dtype=np.uint8)
                train_lbl = identity[train_lbl]

                return train_lbl

            def preprocessing(img_filename,
                              lbl_filename):
                train_img = tf.read_file(self.img_path +
                                         img_filename)
                train_img = tf.image.decode_jpeg(
                    train_img, channels=3)
                train_img = tf.image.resize_images(
                    train_img, [height, width])
                train_img = train_img / 255.0  # Normalize

                train_lbl = tf.py_func(read_labels,
                                       [lbl_filename],
                                       tf.uint8)

                return train_img, train_lbl

            # Input preprocessing
            train_data = train_data.shuffle(
                num_train).map(
                    preprocessing,
                    num_parallel_calls=8).repeat().batch(
                        self.batch_size).prefetch(1)
            valid_data = valid_data.shuffle(
                num_valid).map(
                    preprocessing,
                    num_parallel_calls=8).repeat().batch(
                        self.batch_size).prefetch(1)
            test_data = test_data.shuffle(num_test).map(
                preprocessing,
                num_parallel_calls=8).repeat().batch(
                    self.batch_size).prefetch(1)

            # Create TensorFlow Iterator object
            iterator = train_data.make_initializable_iterator(
            )
            next_element = iterator.get_next()

            # Create initialization ops to switch between the datasets
            self.training_init_op = iterator.make_initializer(
                train_data)
            self.valid_init_op = iterator.make_initializer(
                valid_data)
            self.test_init_op = iterator.make_initializer(
                test_data)

            self.x_train = next_element[0]
            self.y_train = next_element[1]
            self.x_valid = next_element[0]
            self.y_valid = next_element[1]
            self.x_test = next_element[0]
            self.y_test = next_element[1]

            # Training _build_data
            self.num_train_examples = num_train
            self.num_train_batches = \
                (self.num_train_examples + self.batch_size - 1) // \
                self.batch_size
            self.lr_dec_every = self.lr_dec_every * self.num_train_batches

            # Validation _build_data
            self.num_valid_examples = num_valid
            self.num_valid_batches = \
                ((self.num_valid_examples + self.eval_batch_size - 1)
                 // self.eval_batch_size)

            # Test _build_data
            self.num_test_examples = num_test
            self.num_test_batches = \
                ((self.num_test_examples + self.eval_batch_size - 1)
                 // self.eval_batch_size)

    def count_model_params(self, tf_variables):
        """
        Args:
          tf_variables: list of all model variables
        """

        num_vars = 0
        for var in tf_variables:
            num_vars += np.prod(
                [dim.value for dim in var.get_shape()])
        return num_vars

    # Load weight based on layer name
    def create_weight(self,
                      name,
                      shape,
                      initializer=None,
                      trainable=True,
                      seed=None):
        if initializer is None:
            initializer = tf.keras.initializers.he_normal(
                seed=seed)
        return tf.get_variable(name,
                               shape,
                               initializer=initializer,
                               trainable=trainable)

    def batch_norm(self,
                   x,
                   is_training,
                   name="bn",
                   decay=0.9,
                   epsilon=1e-5):
        shape = [x.get_shape()[3]]

        with tf.variable_scope(
                name,
                reuse=None if is_training else True):
            offset = tf.get_variable(
                "offset",
                shape,
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))
            scale = tf.get_variable(
                "scale",
                shape,
                initializer=tf.constant_initializer(
                    1.0, dtype=tf.float32))
            moving_mean = tf.get_variable(
                "moving_mean",
                shape,
                trainable=False,
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))
            moving_variance = tf.get_variable(
                "moving_variance",
                shape,
                trainable=False,
                initializer=tf.constant_initializer(
                    1.0, dtype=tf.float32))

            if is_training:
                x, mean, variance = tf.nn.fused_batch_norm(
                    x,
                    scale,
                    offset,
                    epsilon=epsilon,
                    is_training=True)
                update_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay)
                update_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay)
                with tf.control_dependencies(
                    [update_mean, update_variance]):
                    x = tf.identity(x)
            else:
                x, _, _ = tf.nn.fused_batch_norm(
                    x,
                    scale,
                    offset,
                    mean=moving_mean,
                    variance=moving_variance,
                    epsilon=epsilon,
                    is_training=False)
        return x

    def batch_norm_with_mask(self,
                             x,
                             is_training,
                             mask,
                             num_channels,
                             name="bn",
                             decay=0.9,
                             epsilon=1e-3):
        shape = [num_channels]
        indices = tf.where(mask)
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])

        with tf.variable_scope(
                name,
                reuse=None if is_training else True):
            offset = tf.get_variable(
                "offset",
                shape,
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))
            scale = tf.get_variable(
                "scale",
                shape,
                initializer=tf.constant_initializer(
                    1.0, dtype=tf.float32))
            offset = tf.boolean_mask(offset, mask)
            scale = tf.boolean_mask(scale, mask)

            moving_mean = tf.get_variable(
                "moving_mean",
                shape,
                trainable=False,
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))
            moving_variance = tf.get_variable(
                "moving_variance",
                shape,
                trainable=False,
                initializer=tf.constant_initializer(
                    1.0, dtype=tf.float32))

            if is_training:
                x, mean, variance = tf.nn.fused_batch_norm(
                    x,
                    scale,
                    offset,
                    epsilon=epsilon,
                    is_training=True)
                mean = (1.0 - decay) * \
                       (tf.boolean_mask(moving_mean, mask) - mean)
                variance = (1.0 - decay) * (
                    tf.boolean_mask(moving_variance,
                                    mask) - variance)
                update_mean = tf.scatter_sub(
                    moving_mean,
                    indices,
                    mean,
                    use_locking=True)
                update_variance = tf.scatter_sub(
                    moving_variance,
                    indices,
                    variance,
                    use_locking=True)
                with tf.control_dependencies(
                    [update_mean, update_variance]):
                    x = tf.identity(x)
            else:
                masked_moving_mean = tf.boolean_mask(
                    moving_mean, mask)
                masked_moving_variance = tf.boolean_mask(
                    moving_variance, mask)
                x, _, _ = tf.nn.fused_batch_norm(
                    x,
                    scale,
                    offset,
                    mean=masked_moving_mean,
                    variance=masked_moving_variance,
                    epsilon=epsilon,
                    is_training=False)
        return x

    def get_train_ops(self,
                      loss,
                      tf_variables,
                      train_step,
                      clip_mode=None,
                      grad_bound=None,
                      l2_reg=1e-4,
                      lr_init=0.1,
                      lr_dec_start=0,
                      lr_dec_every=10000,
                      lr_dec_rate=0.1,
                      lr_dec_min=None,
                      lr_cosine=False,
                      lr_max=None,
                      lr_min=None,
                      lr_T_0=None,
                      lr_T_mul=None,
                      num_train_batches=None,
                      optim_algo=None):
        """
        Args:
          clip_mode: "global", "norm", or None.
          moving_average: store the moving average of parameters
        """

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

        if clip_mode is not None:
            assert grad_bound is not None, "Need grad_bound to clip gradients."
            if clip_mode == "global":
                grads, _ = tf.clip_by_global_norm(
                    grads, grad_bound)
            elif clip_mode == "norm":
                clipped = []
                for g in grads:
                    if isinstance(g, tf.IndexedSlices):
                        c_g = tf.clip_by_norm(
                            g.values, grad_bound)
                        c_g = tf.IndexedSlices(
                            g.indices, c_g)
                    else:
                        c_g = tf.clip_by_norm(
                            g, grad_bound)
                    clipped.append(g)
                grads = clipped
            else:
                raise NotImplementedError(
                    "Unknown clip_mode {}".format(
                        clip_mode))

        if lr_cosine:
            assert lr_max is not None, "Need lr_max to use lr_cosine"
            assert lr_min is not None, "Need lr_min to use lr_cosine"
            assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
            assert lr_T_mul is not None, "Need lr_T_mul " \
                                         "to use lr_cosine"
            assert num_train_batches is not None, (
                "Need num_train_batches "
                "to use lr_cosine")

            curr_epoch = train_step // num_train_batches

            last_reset = tf.Variable(0,
                                     dtype=tf.int32,
                                     trainable=False,
                                     name="last_reset")
            T_i = tf.Variable(lr_T_0,
                              dtype=tf.int32,
                              trainable=False,
                              name="T_i")
            T_curr = curr_epoch - last_reset

            def _update():
                update_last_reset = tf.assign(
                    last_reset,
                    curr_epoch,
                    use_locking=True)
                update_T_i = tf.assign(T_i,
                                       T_i * lr_T_mul,
                                       use_locking=True)
                with tf.control_dependencies(
                    [update_last_reset, update_T_i]):
                    rate = tf.to_float(
                        T_curr) / tf.to_float(
                            T_i) * math.pi
                    lr = \
                        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
                return lr

            def _no_update():
                rate = tf.to_float(T_curr) / tf.to_float(
                    T_i) * math.pi
                lr = lr_min + 0.5 * (lr_max - lr_min) * (
                    1.0 + tf.cos(rate))
                return lr

            learning_rate = tf.cond(
                tf.greater_equal(T_curr, T_i), _update,
                _no_update)
        else:
            learning_rate = \
                tf.train.exponential_decay(
                    lr_init,
                    tf.maximum(train_step - lr_dec_start, 0),
                    lr_dec_every,
                    lr_dec_rate, staircase=False)
            if lr_dec_min is not None:
                learning_rate = tf.maximum(
                    learning_rate, lr_dec_min)

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
