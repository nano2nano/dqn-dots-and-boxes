import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


def create_model(state_size, action_size, learning_rate, weights_file_path=None):
    board_input = Input(shape=state_size)
    turn_input = Input(shape=(1, ))

    x = Reshape(state_size+(1, ))(board_input)
    x = Conv2D(filters=256, kernel_size=3,
               strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Model(inputs=board_input, outputs=x)

    x2 = Dense(128, activation='relu')(turn_input)
    x2 = Model(inputs=turn_input, outputs=x2)

    combined = concatenate([x.output, x2.output])
    z = Dense(256, activation='relu')(combined)
    z = Dense(action_size, activation="tanh")(z)

    model = Model(inputs=[board_input, turn_input], outputs=[z])
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=huberloss, optimizer=optimizer)

    if weights_file_path is None:
        print('create new weights')
    else:
        model.load_weights(weights_file_path)
        print('loaded weights file')
    return model


def load_my_model(model_file_path, weights_file_path):
    model = load_model(model_file_path, custom_objects={
                       'huberloss': huberloss})
    model.load_weights(weights_file_path)
    return model
