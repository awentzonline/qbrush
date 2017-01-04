import os

import numpy as np
from keras.models import load_model


class QAgent(object):
    def __init__(self, canvas_shape, num_actions, discount=0.9, name='qagent', ignore_existing=True):
        self.canvas_shape = canvas_shape
        self.num_actions = num_actions
        self.discount = discount
        self.name = name
        self.ignore_existing = ignore_existing
        self.setup_model()

    def setup_model(self):
        if self.ignore_existing or not self.try_load_model():
            self.build_model()

    def build_model(self):
        pass

    def try_load_model(self):
        filename = self.model_filename
        if os.path.exists(filename):
            self.model = load_model(filename, custom_objects=self.model_custom_objects())
            return True
        return False

    def model_custom_objects(self, **kwargs):
        return kwargs

    @property
    def model_filename(self):
        return '{}.h5'.format(self.name)

    def save_model(self):
        self.model.save(self.model_filename)

    def policy(self, states):
        q = self.q(states)
        return np.argmax(q, axis=1)

    def q(self, states):
        return self.model.predict(states)

    def train_epoch(self, num_epochs, num_batches_per_epoch=100, batch_size=32):
        for epoch_i in range(num_epochs):
            for batch_i in tqdm(num_batches_per_epoch):
                state, new_q = self.get_training_batch(batch_size)
                self.model.train_on_batch(state, new_q, verbose=True)

    def train_step(self, s, a, r, s1, t):
        q0 = self.q(s)
        q1 = self.q(s1)
        max_q1 = np.max(q1, axis=1)
        new_q = np.copy(q0)
        new_q[np.arange(new_q.shape[0]), a] = r + t * self.discount * max_q1
        return self.model.train_on_batch(s, new_q)

    def get_training_batch(self, size):
        state, action, reward, next_state = self.sample_memory(size)
        expected_q = self.q(states)  # TODO: get both self.q in one call
        expected_qp = self.q(next_state)
        max_qp = np.max(expected_qp, axis=1)
        new_q = np.copy(expected_q)
        new_q[:, action] = reward + self.discount * max_qp
        return state, new_q

    def sample_memory(self, size):
        return (states, actions, rewards, next_state)
