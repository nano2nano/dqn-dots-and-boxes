import random

import numpy as np


class Agent:
    def __init__(self, main, target, memory, batch_size, gamma, epsilon):
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_size = (9, 13)
        self.main = main
        self.target = target
        self.epsilon = epsilon

    def replay(self):
        if self.memory.len() < self.batch_size:
            return 0

        batchs = self.memory.sample(self.batch_size)
        batch_td_error = 0

        state_b = []
        turn_b = []
        action_b = []
        reward_b = []
        next_state_b = []
        next_turn_b = []
        valid_actions_b = []
        invalid_actions_b = []
        next_valid_actions_b = []
        next_invalid_actions_b = []

        for batch in batchs:
            state_b.append(batch[0])
            turn_b.append(batch[1])
            action_b.append(batch[2])
            reward_b.append(batch[3])
            next_state_b.append(batch[4])
            next_turn_b.append(batch[5])
            valid_actions_b.append(batch[6])
            invalid_actions_b.append(batch[7])
            next_valid_actions_b.append(batch[8])
            next_invalid_actions_b.append(batch[9])
        state_b = np.asarray(state_b).reshape(
            [self.batch_size]+list(self.state_size))
        turn_b = np.asarray(turn_b).reshape([self.batch_size, 1])
        next_state_b = np.asarray(next_state_b).reshape(
            [self.batch_size]+list(self.state_size))
        next_turn_b = np.asarray(next_turn_b).reshape([self.batch_size, 1])

        outputs = self.main.predict([state_b, turn_b], self.batch_size)
        next_state_main_qvals_b = self.main.predict(
            [next_state_b, next_turn_b], self.batch_size)
        next_state_target_qvals_b = self.target.predict(
            [next_state_b, next_turn_b], self.batch_size)

        for i in range(self.batch_size):
            if next_turn_b[i][0] == 0:
                td_error = reward_b[i]
            else:
                best_action_idx = np.argmax(
                    turn_b[i][0] * next_state_main_qvals_b[i][next_valid_actions_b[i]])
                best_action = next_valid_actions_b[i][best_action_idx]
                maxq = next_state_target_qvals_b[i][best_action]
                td_error = reward_b[i] + self.gamma * maxq
            td_error_diff = outputs[i][action_b[i]] - td_error
            if np.abs(td_error_diff) > 0.7:
                print('now : {}'.format(outputs[i][action_b[i]]))
                print('target : {}'.format(td_error))
                print('diff : {}'.format(np.abs(td_error_diff)))
            batch_td_error += np.abs(td_error_diff)
            outputs[i][action_b[i]] = td_error
            self.memory.update(batchs[i], td_error_diff)
        self.main.train_on_batch([state_b, turn_b], np.asarray(outputs))
        return batch_td_error

    def get_action(self, state, turn, valid_actions):
        if self.epsilon <= np.random.uniform(0, 1):
            return self.get_best_action(state, turn, valid_actions)
        else:
            return self.get_random_action(valid_actions)

    def get_random_action(self, valid_actions):
        return random.choice(valid_actions)

    def get_best_action(self, state, turn, valid_actions):
        state = np.asarray(state).reshape((1, ) + self.state_size)
        turn = np.asarray(turn).reshape((1, 1))
        q_values = turn[0] * self.main.predict([state, turn])[0]
        best_action_idx = np.argmax(q_values[valid_actions])
        return valid_actions[best_action_idx]

    def update_target_weights(self):
        self.target.set_weights(self.main.get_weights())

    def save_weights(self, file_path):
        self.main.save_weights(file_path)

    def save_memory(self, file_path):
        self.memory.save(file_path)

    def save_model(self, file_path):
        self.main.save(file_path)

    def save_params(self, model_file_path, weights_file_path, memory_file_path):
        self.save_model(model_file_path)
        self.save_weights(weights_file_path)
        self.save_memory(memory_file_path)
