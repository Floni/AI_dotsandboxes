from collections import deque

from segment_tree import MinSegmentTree, SumSegmentTree

import random

import numpy as np

class ReplayMemory:
    def __init__(self, max_mem):
        self.memory = []
        self.next_idx = 0
        self.max_mem = max_mem

    def add_mem(self, state, action, next_state, reward, done, next_state_valid):  #state_valid, next_state_valid):
        data = (state, action, next_state, reward, done, next_state_valid)
        if self.next_idx >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.max_mem

    def get_batch(self, size):
        return random.sample(self.memory, k=size)

class PriorityReplayMemory(ReplayMemory):
    def __init__(self, size, alpha):
        super().__init__(size)
        self.alpha = alpha

        tree_cap = 1
        while tree_cap < size:
            tree_cap *= 2

        self.sum_tree = SumSegmentTree(tree_cap)
        self.max_priority = 1.0

    def add_mem(self, *args, **kwargs):
        idx = self.next_idx
        super().add_mem(*args, **kwargs)
        prio = self.max_priority ** self.alpha
        self.sum_tree[idx] = prio

    def _sample_idxs(self, size):
        idxs = []
        for _ in range(size):
            mass = random.random() * self.sum_tree.sum(0, len(self.memory)-1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)
        return idxs

    def get_batch(self, size, beta):
        assert beta > 0
        idxs = self._sample_idxs(size)

        # create weights:
        weights = []
        tot_sum = self.sum_tree.sum()

        #p_min = tot_min / tot_sum
        #max_w = (p_min * len(self.memory)) ** (-beta)

        for idx in idxs:
            p_sample = self.sum_tree[idx] / tot_sum
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight)

        #print(max_w)
        #print(weights)
        weights = np.array(weights)
        weights /= max(weights)

        batch = [self.memory[idx] for idx in idxs]
        return batch, weights, idxs

    def update_priorities(self, idxs, priorities):
        for idx, prio in zip(idxs, priorities):
            assert prio > 0
            prio_a = prio ** self.alpha
            self.sum_tree[idx] = prio_a

            self.max_priority = max(self.max_priority, prio)
