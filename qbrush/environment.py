import os

import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
from tqdm import tqdm

from .argparse_utils import CommaSplitAction
from .image_utils import save_image_grid


class QBrushEnvironment(object):
    actions = []

    def __init__(self, config, num_canvases=1):
        self.num_canvases = num_canvases
        self.config = config
        self._prepare_vgg()
        self.reset()

    def reset(self):
        if self.config.channels == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        self.canvases = [
            Image.new(mode, self.image_size, self.config.blank_color)
            for _ in range(self.num_canvases)
        ]
        self.image_arr = np.zeros((self.num_canvases,) + self.image_shape)
        self.update_image_array()
        self.is_complete = False

    def _prepare_vgg(self):
        vgg = vgg16.VGG16(include_top=False, input_shape=self.vgg_image_shape)
        outputs = []
        for layer_name in self.config.feature_layers:
            layer = vgg.get_layer(layer_name)
            outputs.append(layer.output)
        self.vgg_features = Model(vgg.inputs, outputs)
        self.vgg_features.compile(optimizer='adam', loss='mse')

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def image_size(self):
        return (self.config.width, self.config.height)

    @property
    def image_shape(self):
        if K.image_dim_ordering() == 'tf':
            return (self.config.height, self.config.width, self.config.channels)
        else:
            return (self.config.channels, self.config.height, self.config.width)

    @property
    def vgg_image_shape(self):
        if K.image_dim_ordering() == 'tf':
            return (self.config.height, self.config.width, 3)
        else:
            return (3, self.config.height, self.config.width)

    @property
    def image_shape_dims(self):
        return (self.config.height, self.config.width)

    def update_image_array(self):
        for canvas_i, canvas in enumerate(self.canvases):
            self.image_arr[canvas_i] = img_to_array(canvas)
        #self.image_features = self.get_image_features(self.image_arr)

    def get_image_features(self, images):
        if K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            axis = 1
        if images.shape[axis] == 1:
            images = np.repeat(images, 3, axis=axis)
        images = vgg16.preprocess_input(images)
        return self.vgg_features.predict(images)

    def simulate(self, agent, max_steps=1000, epsilon=0.5, train_p=0.0):
        self.is_complete = False
        last_state = None
        history = []
        for step_i in tqdm(range(max_steps)):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, self.num_actions, (self.num_canvases,))
            else:
                action = agent.policy(self.get_state())
            self.perform_action(action)
            self.update_image_array()
            reward = self.calculate_reward()
            this_state = self.get_state()
            if last_state and train_p and (reward.any() >= 0. or np.random.uniform(0., 1.) < train_p):
                loss = agent.train_step(last_state, action, reward, this_state)
                history.append(loss)
            last_state = this_state
            if np.random.uniform(0., 1.) < 0.1:
                self.save_image_state('output.png')
            if self.is_complete:
                break
        return history

    def perform_action(self, actions):
        for canvas_i in range(self.num_canvases):
            action_name = self.actions[actions[canvas_i]]
            getattr(self, 'perform_{}'.format(action_name))(canvas_i)

    def get_state(self):
        return self.image_arr

    def calculate_reward(self):
        return np.array([-1] * self.num_canvases)

    def save_image_state(self, filename):
        save_image_grid(
            self.canvases, os.path.join(self.config.output_path, filename)
        )

    @classmethod
    def add_to_arg_parser(cls, parser):
        parser.add_argument('--width', type=int, default=64)
        parser.add_argument('--height', type=int, default=64)
        parser.add_argument('--channels', type=int, default=3)
        parser.add_argument('--blank-color', default='black')
        parser.add_argument(
            '--feature-layers', type=CommaSplitAction,
            default=['block4_conv1']
        )
