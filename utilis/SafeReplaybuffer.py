import pickle
import random

import numpy as np


class SafeReplayMemory:
    def __init__(self, capacity, seed, recent_fraction=0.2):
        random.seed(seed)
        self.capacity = capacity
        self.recent_fraction = float(recent_fraction)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, cost, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if self.recent_fraction <= 0.0:
            return self.sample_uniform(batch_size)

        num_recent = int(round(batch_size * self.recent_fraction))
        num_recent = min(max(num_recent, 0), batch_size)
        num_uniform = batch_size - num_recent

        uniform_indices = random.sample(range(buffer_len), num_uniform)

        recent_window = min(2048, buffer_len)
        if self.position - recent_window >= 0:
            recent_indices_raw = list(range(self.position - recent_window, self.position))
        else:
            recent_indices_raw = (
                list(range(self.position - recent_window + buffer_len, buffer_len))
                + list(range(0, self.position))
            )

        recent_count = min(num_recent, len(recent_indices_raw))
        recent_indices = random.sample(recent_indices_raw, recent_count)
        if recent_count < num_recent:
            uniform_indices += random.sample(range(buffer_len), num_recent - recent_count)

        all_indices = uniform_indices + recent_indices
        batch = [self.buffer[idx] for idx in all_indices]

        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def sample_uniform(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path, i_episode):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
