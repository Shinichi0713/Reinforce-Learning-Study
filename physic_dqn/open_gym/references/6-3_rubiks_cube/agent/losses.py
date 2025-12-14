import tensorflow as tf
from tensorflow.keras import backend as K


# huber loss for value
def huber_loss(y_true, y_pred, huber_loss_delta=10.0):
    err = y_true - y_pred

    cond = K.abs(err) < huber_loss_delta
    L2 = 0.5 * K.square(err)
    L1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)

    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


def _clip_value(value, epsilon=1.0E-08):
    return K.clip(value, epsilon, 1.0 - epsilon)


# entropy term
def _calc_entropy(y_pred):
    # actually minus entropy
    logp = K.log(_clip_value(y_pred))

    return K.sum(y_pred * logp, axis=1)


# policy gradient loss for policy
def policy_gradient_loss(y_true, td_err, y_pred, entropy_lambda=1.0E-03):
    prj_prob = K.sum(y_true * y_pred, axis=1)
    logp_prj = K.log(_clip_value(prj_prob))
    loss = -1.0 * td_err * logp_prj

    loss += entropy_lambda * _calc_entropy(y_pred)

    return K.mean(loss)
