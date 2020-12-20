import numpy as np
import random


class DotsAndBoxes():
    VERSION = '1.1'
    ROWS = 5
    COLUMNS = 7

    def __init__(self):
        super().__init__()
        self.state_size = 2*self.ROWS-1, 2*self.COLUMNS-1
        self.actions = self._create_actions()
        self.action_size = len(self.actions)
        self.reset()

    def reset(self):
        self.done = False
        self.winner = 0
        self.foul = False
        self.score = [0, 0]  # -1, 1
        self.turn = -1
        self.board = np.zeros(self.state_size)
        self.available_actions = list(range(self.action_size))
        self.invalid_actions = list()

        return self.observe(), self._get_reward, self.done, {}

    def step(self, action):
        player = self.turn
        if self._is_valid_action(action):
            self.available_actions.remove(action)
            self.invalid_actions.append(action)
            i, j = self.actions[action]
            self.board[i, j] = player
            self._update(player, i, j)  # change turn if did not fill any box
            if len(self.available_actions) == 0:
                self._close_game('done', player)
        else:
            self._close_game('foul', player)

        return self.observe(), self._get_reward(), self.done, {}

    def observe(self):
        return np.array([self.board])

    def _close_game(self, game_state, player):
        if self.winner != 0:
            raise ValueError('Invalid winner')
        if game_state == 'foul':
            self.winner = -player
            self.foul = True
        elif game_state == 'done':
            if self.score[0] == self.score[1]:
                # drow
                self.winner = 0
            else:
                self.winner = -1 if self.score[0] > self.score[1] else 1
        self.done = True
        self.turn = 0

    def _get_reward(self):
        return self.winner

    def _update(self, player, i, j):
        vertical = i % 2 > j % 2
        a = self._check_box(
            player, i, j-1) if vertical else self._check_box(player, i-1, j)
        b = self._check_box(
            player, i, j+1) if vertical else self._check_box(player, i+1, j)
        if not (a or b):  # Did not fill any box
            self.turn *= -1

    def _check_box(self, player, i, j):
        hight, width = self.board.shape
        if (i and i < hight) and (j and j < width):
            filled = self.board[i-1, j] and self.board[i+1,
                                                       j] and self.board[i, j-1] and self.board[i, j+1]
            if filled:
                self.board[i, j] = player
                self.score[player == 1] += 1
            return filled

    def _is_valid_action(self, action):
        return action in self.available_actions

    def _create_actions(self):
        edges = np.reshape([i % 2 != j % 2 for j in range(2*self.COLUMNS-1)
                            for i in range(2*self.ROWS-1)], (2*self.ROWS-1, 2*self.COLUMNS-1))
        actions = np.where(edges)
        return list(zip(actions[0], actions[1]))
