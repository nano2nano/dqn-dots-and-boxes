import os
import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (BatchNormalization, Conv2D, Dense, Flatten, Input,
                          LeakyReLU, ReLU, Reshape, concatenate)
from keras.models import Model, model_from_json

from dots_and_boxes import DotsAndBoxes


class QNetwork:
    def __init__(self):
        self.action_size = 58
        self.state_size = (9, 13)
        self.actions = set(range(58))

        board_input = Input(shape=self.state_size)
        turn_input = Input(shape=(1, ))

        x = Reshape(self.state_size+(1, ))(board_input)
        x = Conv2D(filters=128, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(1024)(x)
        x = LeakyReLU()(x)
        x = Model(inputs=board_input, outputs=x)

        x2 = Dense(128)(turn_input)
        x2 = LeakyReLU()(x2)
        x2 = Dense(128)(x2)
        x2 = LeakyReLU()(x2)
        x2 = Model(inputs=turn_input, outputs=x2)

        combined = concatenate([x.output, x2.output])
        z = Dense(self.action_size, activation="linear")(combined)

        self.model = Model(inputs=[board_input, turn_input], outputs=[z])
        self.model.load_weights('./params/weights.h5')

    def _load_model(self, path, filename):
        with open(os.path.join(path, filename)) as f:
            json_string = f.read()
            model = model_from_json(json_string)
        return model

    def get_best_action(self, state, turn):
        q_values = self.model.predict([state, turn])[0]
        q_values = turn * q_values
        return np.argmax(q_values)


def vs_random(network, battle_num=100):
    win = 0
    lose = 0
    drow = 0
    env = DotsAndBoxes()
    for i in range(battle_num):
        is_agent = False
        next_state, _, done, _ = env.reset()
        agent_turn = random.choice([-1, 1])
        t = 0
        while not done:
            if 58-len(env.available_actions) != t:
                raise ValueError('available_actionが不正です')
            t += 1
            # 状態更新
            state = next_state
            # 行動選択
            is_agent = (env.turn == agent_turn)
            if is_agent:
                action = network.get_best_action(state, np.array([env.turn]))
            else:
                action = random.choice(list(env.available_actions))
            next_state, _, done, _ = env.step(action)
        # ゲーム終了
        if env.winner == agent_turn:
            win += 1
        elif env.winner == -1*agent_turn:
            lose += 1
        elif env.winner == 0:
            drow += 1
        else:
            raise ValueError('winnerに予期せぬ値がセットされています')
    return win, lose, drow


if __name__ == "__main__":
    win, lose, drow = vs_random(QNetwork())
    print('win : %d , lose : %d , drow : %d' % (win, lose, drow))
