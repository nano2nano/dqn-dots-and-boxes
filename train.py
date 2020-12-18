import datetime
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collections import deque

import numpy as np
from dots_and_boxes import DotsAndBoxes

from model import create_model
from per_rank_base_memory import PERRankBaseMemory


class QNetwork:
    def __init__(self, state_size=(9, 13), action_size=58, learning_rate=None, weights_path=None):
        self.action_size = action_size
        self.state_size = state_size
        self.actions = set(range(58))

        self.model = create_model(self.state_size, self.action_size,
                                  learning_rate=learning_rate, weights_path=weights_path)

    def replay(self, memory, batch_size, gamma, targetQN):
        indexes, batchs, _ = memory.sample(batch_size)
        batch_td_error = 0

        state_b = []
        turn_b = []
        action_b = []
        reward_b = []
        next_state_b = []
        next_turn_b = []
        available_actions_b = []
        invalid_actions_b = []

        for batch in batchs:
            state_b.append(batch[0])
            turn_b.append(batch[1])
            action_b.append(batch[2])
            reward_b.append(batch[3])
            next_state_b.append(batch[4])
            next_turn_b.append(batch[5])
            available_actions_b.append(batch[6])
            invalid_actions_b.append(batch[7])
        state_b = np.asarray(state_b).reshape([batch_size, 9, 13])
        turn_b = np.asarray(turn_b).reshape([batch_size, 1])
        next_state_b = np.asarray(next_state_b).reshape([batch_size, 9, 13])
        next_turn_b = np.asarray(next_turn_b).reshape([batch_size, 1])

        outputs = self.model.predict([state_b, turn_b], batch_size)
        state1_model_qvals_b = self.model.predict(
            [next_state_b, next_turn_b], batch_size)
        state1_target_qvals_b = targetQN.model.predict(
            [next_state_b, next_turn_b], batch_size)

        for i in range(batch_size):
            if next_turn_b[i][0] == 0:
                td_error = reward_b[i]
            else:
                maxq = state1_target_qvals_b[i][np.argmax(
                    turn_b[i][0] * state1_model_qvals_b[i])]
                td_error = reward_b[i] + gamma * maxq
            td_error_diff = outputs[i][action_b[i]] - td_error
            cond = np.abs(td_error_diff) > 0.7 and next_turn_b[i][0] == 0
            if cond:
                print('now : {}'.format(outputs[i][action_b[i]]))
                print('target : {}'.format(td_error))
                print('diff : {}'.format(np.abs(td_error_diff)))
            batch_td_error += np.abs(td_error_diff)
            outputs[i][invalid_actions_b[i]] = -turn_b[i]
            outputs[i][action_b[i]] = td_error
            memory.update(batchs[i], td_error_diff)
        self.model.train_on_batch([state_b, turn_b], np.asarray(outputs))
        return batch_td_error

    def get_best_action(self, state, turn, valid_actions=None):
        q_values = self.model.predict([state, turn])[0]
        q_values = turn[0] * q_values
        if valid_actions is None:
            return np.argmax(q_values)
        else:
            idx = np.argmax(q_values[valid_actions])
            return valid_actions[idx]


if __name__ == "__main__":
    PARAMS_DIR = './params'
    MODEL_FILE = PARAMS_DIR+'/model.json'
    DQN_MODE = 0  # 1がDQN、0がDDQN

    env = DotsAndBoxes()
    num_episodes = 100000  # 総試行回数
    gamma = 0.99
    isrender = 0

    learning_rate = 1e-5
    memory_capacity = 1_000_000
    batch_size = 32

    per_alpha = 0.6
    per_beta_initial = 0.0
    per_beta_steps = 100_000
    per_enable_is = False

    state_size = env.observe().shape

    mainQN = QNetwork(learning_rate=learning_rate, action_size=env.action_size)
    targetQN = QNetwork(learning_rate=learning_rate,
                        action_size=env.action_size)
    json_string = mainQN.model.to_json()
    os.makedirs(PARAMS_DIR, exist_ok=True)
    with open(MODEL_FILE, mode='w') as f:
        f.write(json_string)
    memory = PERRankBaseMemory(memory_capacity, per_alpha,
                               per_beta_initial, per_beta_steps, MODEL_FILE)
