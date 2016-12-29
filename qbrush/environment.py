import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
from tqdm import tqdm

from .argparse_utils import CommaSplitAction


class QBrushEnvironment(object):
    actions = []

    def __init__(self, config):
        self.config = config
        self._prepare_vgg()
        self.reset()

    def reset(self):
        self.image = Image.new('RGB', self.image_size, self.config.blank_color)
        self.update_image_array()
        self.is_complete = False

    def _prepare_vgg(self):
        vgg = vgg16.VGG16(include_top=False, input_shape=self.image_shape)
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
            return (self.config.height, self.config.width, 3)
        else:
            return (3, self.config.height, self.config.width)

    def update_image_array(self):
        self.image_arr = img_to_array(self.image)
        #self.image_features = self.get_image_features(self.image_arr)

    def get_image_features(self, image):
        return self.vgg_features.predict(image[None, ...])

    def simulate(self, agent, max_steps=1000, epsilon=0.5):
        self.is_complete = False
        last_state = None
        for step_i in tqdm(range(max_steps)):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                action = agent.policy(self.get_state())[0]
            self.perform_action(action)
            self.update_image_array()
            reward = self.calculate_reward()
            this_state = self.get_state()
            if last_state and np.random.uniform(0., 1.) < 0.5:
                agent.train_step(last_state, action, reward, this_state)
            last_state = this_state
            if np.random.uniform(0., 1.) < 0.1:
                self.save_image_state('output.png')
            if self.is_complete:
                break

    def perform_action(self, action):
        action_name = self.actions[action]
        getattr(self, 'perform_{}'.format(action_name))()

    def get_state(self):
        return self.image_arr

    def calculate_reward(self):
        return -1

    def save_image_state(self, filename):
        self.image.save(filename)

    @classmethod
    def add_to_arg_parser(cls, parser):
        parser.add_argument('--width', type=int, default=64)
        parser.add_argument('--height', type=int, default=64)
        parser.add_argument('--blank-color', default='black')
        parser.add_argument(
            '--feature-layers', type=CommaSplitAction,
            default=['block4_conv1']
        )
