import bisect
import random

import numpy as np


class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0

    def __lt__(self, o):  # a<b
        return self.priority > o.priority


class PERRankBaseMemory():
    def __init__(self, capacity, alpha, memory_path=None):
        self.capacity = capacity
        self.memory_path = memory_path
        try:
            self.buffer = np.load(self.memory_path+'.npy',
                                  allow_pickle=True).tolist()
            print("LoadMemorySize：", self.len())
        except:
            self.buffer = []
        self.alpha = alpha

        self.max_priority = 1

    def save(self, file_path):
        np.save(file_path, self.buffer)

    def len(self):
        return len(self.buffer)

    def add(self, experience):
        if self.capacity <= self.len():
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()

        experience = _bisect_wrapper(experience)
        experience.priority = self.max_priority
        bisect.insort(self.buffer, experience)

    def update(self, experience, td_error):
        priority = (abs(td_error) + 0.0001)  # priority を計算

        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size):
        indexes = []
        batchs = []

        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section*i + random.random()*section)

        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

        return batchs
