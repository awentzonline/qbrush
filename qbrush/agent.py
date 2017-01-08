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
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.train_target_model(tau=1.0)  # copy the random initial weights
        print self.model.summary()

    def build_model(self):
        pass

    def try_load_model(self):
        filename = self.model_filename
        if os.path.exists(filename):
            self.model = load_model(filename, custom_objects=self.model_custom_objects())
            self.target_model = load_model(filename, custom_objects=self.model_custom_objects())
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

    def q(self, states, train=False):
        if train:
            model = self.target_model
        else:
            model = self.model
        return model.predict(states)

    def train_step(self, s, a, r, s1, t):
        q0 = self.q(s, train=True)
        q1 = self.q(s1, train=True)
        max_q1 = np.max(q1, axis=1)
        new_q = np.copy(q0)
        new_q[np.arange(new_q.shape[0]), a] = r + np.logical_not(t) * self.discount * max_q1
        return self.model.train_on_batch(s, new_q)

    def train_target_model(self, tau=0.001):
        '''Good article: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html'''
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
