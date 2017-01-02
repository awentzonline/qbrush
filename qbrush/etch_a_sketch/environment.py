import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw

from qbrush.environment import QBrushEnvironment
from qbrush.image_utils import save_image_array_grid


class EtchASketchEnvironment(QBrushEnvironment):
    actions = [
        'move_up',
        'move_down',
        'move_left',
        'move_right',
    ]
    move_size = 0.05

    def update_targets(self, target_images):
        self.target_images = target_images
        self.target_features = self.get_image_features(target_images)
        self.update_canvas_error()

    def mse(self, a, b):
        return np.square(a - b).sum(axis=(1, 2, 3))

    def update_canvas_error(self):
        canvas_features = self.get_image_features(self.image_arr)
        self.canvas_error = self.mse(self.target_features, canvas_features)

    def reset(self):
        super(EtchASketchEnvironment, self).reset()
        self.position = np.random.uniform(0., 1., (self.num_canvases, 2))
        self.position_maps = np.zeros((self.num_canvases,) + self.image_shape_dims)
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
        draw.line(
            [tuple(start * image_size), tuple(self.position[canvas_id] * image_size)],
            self.config.color)
        self._update_position_maps()

    def get_state(self):
        return [
            self._expanded_position_maps(),
            self.image_arr,
            self.target_images,
        ]

    def _expanded_position_maps(self):
        axis = -1
        if K.image_dim_ordering() == 'th':
            axis = 1
        pms = np.expand_dims(self.position_maps, axis)
        return pms * 255.

    def _update_position_maps(self):
        self.position_maps *= self.config.position_decay
        grid_shape = np.array(self.position_maps.shape[1:]) - 1
        indexes = (self.position[:, ::-1] * grid_shape).astype(np.int32)
        indexes = tuple(indexes.transpose((1,0)))
        indexes = (np.arange(self.position_maps.shape[0]),) + indexes
        self.position_maps[indexes] = 1.
        if np.random.uniform(0., 1.) < 0.:
            axis = -1
            if K.image_dim_ordering() == 'th':
                axis = 1
            pms = np.repeat(self._expanded_position_maps(), 3, axis=axis)
            samples = np.concatenate([pms, self.image_arr + 120.], axis=0)
            save_image_array_grid(samples, 'positions.png')

    def calculate_reward(self):
        canvas_features = self.get_image_features(self.image_arr)
        err = np.square(self.target_features - canvas_features).sum(axis=(1, 2, 3))
        reward = np.array([-1.] * self.num_canvases).astype(np.float32)
        if not self.last_err is None:
            reward[err < self.last_err] = 1.
        self.last_err = err
        if self.terminal:
            err_ratio = self.canvas_error / (err + 1e-7)
            updates = err_ratio * 100.
            updates[err_ratio < 1] = -50.
            reward += updates
        return reward

    @classmethod
    def add_to_arg_parser(cls, parser):
        super(EtchASketchEnvironment, cls).add_to_arg_parser(parser)
        parser.add_argument('--position-decay', type=float, default=0.8)
        parser.add_argument('--color', default='white')
