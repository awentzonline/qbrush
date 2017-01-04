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


class Environment(object):
    actions = []

    def __init__(self, config, num_actors=1):
        self.config = config
        self.num_actors = num_actors
        self.setup()

    def setup(self):
        pass

    def reset(self):
        for i in range(self.num_actors):
            self.reset_actor(i)

    def reset_actor(self, actor_i):
        pass

    def step(self, action):
        info = {}
        self.reset_terminal_actors()
        self.pre_action(info)
        self.perform_action(action)
        self.post_action(info)
        reward = self.reward()
        new_state = self.get_state()
        is_terminal = self.is_terminal()
        return new_state, reward, is_terminal, info

    def reset_terminal_actors(self):
        terminal_actors = (self.is_terminal() * range(1, self.num_actors + 1)).nonzero()[0]
        for i in range(terminal_actors.shape[0]):
            actor_i = terminal_actors[i]
            self.reset_actor(actor_i)

    def pre_action(self, info):
        pass

    def perform_action(self, action):
        for actor_i in range(self.num_actors):
            action_name = self.actions[action[actor_i]]
            getattr(self, 'perform_{}'.format(action_name))(actor_i)

    def post_action(self, info):
        pass

    def get_state(self):
        return []

    def reward(self):
        return np.array([-1.] * self.num_actors)

    def is_terminal(self):
        return np.array([False] * self.num_actors)

    @property
    def num_actions(self):
        return len(self.actions)

    @classmethod
    def add_to_arg_parser(cls, parser):
        pass


class QBrushEnvironment(Environment):

    def setup(self):
        self._prepare_vgg()
        # create blank canvases
        if self.config.channels == 3:
            self.canvas_mode = 'RGB'
        else:
            self.canvas_mode = 'L'
        self.canvases = [None] * self.num_actors
        self.image_arr = np.zeros((self.num_actors,) + self.image_shape)

    def reset_actor(self, actor_i):
        self.canvases[actor_i] = Image.new(
            self.canvas_mode, self.image_size, self.config.blank_color
        )
        self.update_actor_image_array(actor_i)

    def update_image_array(self):
        for actor_i in range(self.num_actors):
            self.update_actor_image_array(actor_i)

    def update_actor_image_array(self, actor_i):
        self.image_arr[actor_i] = img_to_array(self.canvases[actor_i])

    def _prepare_vgg(self):
        vgg = vgg16.VGG16(include_top=False, input_shape=self.vgg_image_shape)
        outputs = []
        for layer_name in self.config.feature_layers:
            layer = vgg.get_layer(layer_name)
            outputs.append(layer.output)
        self.vgg_features = Model(vgg.inputs, outputs)
        self.vgg_features.compile(optimizer='adam', loss='mse')

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

    def get_image_features(self, images):
        if K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            axis = 1
        if images.shape[axis] == 1:
            images = np.repeat(images, 3, axis=axis)
        images = vgg16.preprocess_input(images)
        return self.vgg_features.predict(images)

    def post_action(self, info):
        self.update_image_array()

    def get_state(self):
        return self.image_arr

    def save_canvas_state(self, filename):
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
