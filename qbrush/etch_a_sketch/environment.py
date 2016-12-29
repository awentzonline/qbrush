import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw

from qbrush.environment import QBrushEnvironment


class EtchASketchEnvironment(QBrushEnvironment):
    actions = [
        'move_up',
        'move_down',
        'move_left',
        'move_right',
    ]
    move_size = 0.05

    def __init__(self, target_image, *args, **kwargs):
        super(EtchASketchEnvironment, self).__init__(*args, **kwargs)
        self.target_image = target_image
        self.target_features = self.get_image_features(target_image)

    def reset(self):
        super(EtchASketchEnvironment, self).reset()
        self.position = np.random.uniform(0., 1., (2,))
        self.last_err = float('inf')

    def perform_move_up(self):
        self._perform_move(0., -self.move_size)

    def perform_move_down(self):
        self._perform_move(0., self.move_size)

    def perform_move_left(self):
        self._perform_move(-self.move_size, 0.)

    def perform_move_right(self):
        self._perform_move(self.move_size, 0.)

    def _perform_move(self, dx, dy):
        start = np.copy(self.position)
        self.position[0] += dx
        self.position[1] += dy
        self.position = self.position.clip(0.0, 1.0)
        image_size = self.image.size
        draw = ImageDraw.Draw(self.image)
        draw.line([tuple(start * image_size), tuple(self.position * image_size)])

    def get_state(self):
        return [
            self.position[None, ...],
            self.image_arr[None, ...],
            self.target_image[None, ...],
        ]

    def calculate_reward(self):
        canvas_features = self.get_image_features(self.image_arr)
        err = np.square(self.target_features - canvas_features).sum()
        reward = -1.
        if not self.last_err is None and err < self.last_err:
            reward = 1.
        self.last_err = err
        return reward
