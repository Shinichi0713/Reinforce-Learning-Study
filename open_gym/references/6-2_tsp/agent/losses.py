import tensorflow as tf
from tensorflow.keras import backend as K


# huber loss for value
def rms_loss(y_true, y_pred, huber_loss_delta=1.0):
    err = y_true - y_pred

    cond = K.abs(err) < huber_loss_delta
    l2 = 0.5 * K.square(err)
    l1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)

    loss = tf.where(cond, l2, l1)

    return K.mean(loss)


# policy gradient loss for policy
def policy_gradient_loss(reward, log_probs):
    loss = -1.0 * reward * log_probs

    return K.mean(loss)


# for distance tensor calculation
def tour_distance(input_data, tour):
    batch_size = tf.shape(input_data)[0]
    seq_length = tf.shape(input_data)[1]

    # sort input_data by tour

    # obtain permutation indices
    # [n_batch, 1]
    _batch_seq = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), 1)
    # [n_batch*1, 1*n_seq']
    # n_seq' = n_seq+1; because end (=start) location explicitly included
    _seq_extend = tf.tile(_batch_seq, [1, seq_length + 1])
    # [n_batch*1, n_seq', 2]
    _perm_indices = tf.stack([_seq_extend, tour], axis=2)

    # sort input_data based on the 2-D permutation indices
    _sorted_input = tf.gather_nd(input_data, _perm_indices)
    # [n_batch, n_seq', n_dim] to [n_dim, n_seq', n_batch]
    _sorted_input = tf.transpose(_sorted_input, [2, 1, 0])

    # ordered coordinates
    # [n_seq', n_batch]
    _sorted_x = _sorted_input[0]
    _sorted_y = _sorted_input[1]

    # euclidean inter distance btw locations
    # [n_batch, n_seq']
    dlt_x2 = tf.transpose(tf.square(_sorted_x[1:] - _sorted_x[:-1]), [1, 0])
    dlt_y2 = tf.transpose(tf.square(_sorted_y[1:] - _sorted_y[:-1]), [1, 0])
    inter_distances = tf.sqrt(dlt_x2 + dlt_y2)

    # total tour distance
    # [n_batch]
    tour_dist = tf.reduce_sum(inter_distances, axis=1)

    return tour_dist
