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

    def __init__(self, target_images, *args, **kwargs):
        kwargs['num_canvases'] = target_images.shape[0]
        super(EtchASketchEnvironment, self).__init__(*args, **kwargs)
        self.target_images = target_images
        self.target_features = self.get_image_features(target_images)

    def reset(self):
        super(EtchASketchEnvironment, self).reset()
        self.position = np.random.uniform(0., 1., (self.num_canvases, 2))
        self.last_err = None

    def perform_move_up(self, canvas_id):
        self._perform_move(canvas_id, 0., -self.move_size)

    def perform_move_down(self, canvas_id):
        self._perform_move(canvas_id, 0., self.move_size)

    def perform_move_left(self, canvas_id):
        self._perform_move(canvas_id, -self.move_size, 0.)

    def perform_move_right(self, canvas_id):
        self._perform_move(canvas_id, self.move_size, 0.)

    def _perform_move(self, canvas_id, dx, dy):
        start = np.copy(self.position[canvas_id])
        canvas = self.canvases[canvas_id]
        self.position[canvas_id][0] += dx
        self.position[canvas_id][1] += dy
        self.position[canvas_id] = self.position[canvas_id].clip(0.0, 1.0)
        image_size = np.array(canvas.size) - 1
        draw = ImageDraw.Draw(canvas)
        draw.line([tuple(start * image_size), tuple(self.position[canvas_id] * image_size)])

    def get_state(self):
        return [
            self.position,
            self.image_arr,
            self.target_images,
        ]

    def calculate_reward(self):
        canvas_features = self.get_image_features(self.image_arr)
        err = np.square(self.target_features - canvas_features).sum(axis=(1, 2, 3))
        reward = np.array([-1.] * self.num_canvases).astype(np.float32)
        if not self.last_err is None:
            reward[err < self.last_err] = 1.
        self.last_err = err
        return reward
