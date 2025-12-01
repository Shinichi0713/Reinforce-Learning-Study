from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization


def build_model(input_shape, val_shape, act_shape, n_neurons=512):

    num_value = val_shape
    num_action = act_shape

    input_state = Input(shape=(*input_shape,), name='input_state')
    x = Flatten()(input_state)

    x = Dense(n_neurons, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(n_neurons/4, activation='relu')(x)
    hidden = BatchNormalization()(x)

    x_value = Dense(n_neurons/8, activation='relu')(hidden)
    x_value = BatchNormalization()(x_value)
    x_value = Dense(n_neurons/8, activation='relu')(x_value)
    x_value = BatchNormalization()(x_value)

    value = Dense(num_value, activation='linear',
                  name='output_value')(x_value)

    x_action = Dense(n_neurons/8, activation='relu')(hidden)
    x_action = BatchNormalization()(x_action)
    x_action = Dense(n_neurons/8, activation='relu')(x_action)
    x_action = BatchNormalization()(x_action)

    action = Dense(num_action, activation='softmax',
                   name='output_action')(x_action)

    critic_model = Model(inputs=input_state, outputs=value)
    actor_model = Model(inputs=input_state, outputs=action)

    return critic_model, actor_model
