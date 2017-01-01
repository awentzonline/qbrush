from keras import backend as K
from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, Input, LeakyReLU, merge
)
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from qbrush.advantage import AdvantageAggregator
from qbrush.agent import QAgent


class EtchASketchAgent(QAgent):
    def build_model(self):
        kernel_size = 3
        # input: position, canvas, target
        p = position_in = Input(shape=self.position_shape)
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([position_in, canvas_in, target_in], mode='concat')
        x = Convolution2D(
            32, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(
            64, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(
            128, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        y = x = Flatten()(x)
        # combine
        #y = merge([p, x], mode='concat')
        y = Dense(256)(y)
        y = Dropout(0.3)(y)
        y = LeakyReLU()(y)
        y = Dense(self.num_actions)(y)
        self.model = Model([position_in, canvas_in, target_in], y)
        optimizer = RMSprop(lr=0.0000625, rho=0.99, clipnorm=10.)
        self.model.compile(optimizer=optimizer, loss='mse')
        print self.model.summary()

    @property
    def position_shape(self):
        if K.image_dim_ordering() == 'tf':
            return self.canvas_shape[:-1] + (1,)
        else:
            return (1,) + self.canvas_shape[1:]


class EtchASketchAdvantageAgent(EtchASketchAgent):
    def build_model(self):
        kernel_size = 3
        # input: position, canvas, target
        p = position_in = Input(shape=self.position_shape)
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([position_in, canvas_in, target_in], mode='concat')
        x = Convolution2D(
            32, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(
            64, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(
            128, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        # dueling advantage learning
        v = Dense(256)(x)
        v = Dropout(0.3)(v)
        v = LeakyReLU()(v)
        v = Dense(self.num_actions)(v)
        a = Dense(256)(x)
        a = Dropout(0.3)(a)
        a = LeakyReLU()(a)
        a = Dense(self.num_actions)(a)
        a = AdvantageAggregator()(a)

        q = merge([v, a], mode='sum')
        self.model = Model([position_in, canvas_in, target_in], q)
        optimizer = RMSprop(lr=0.0000625, rho=0.99, clipnorm=10.)
        self.model.compile(optimizer=optimizer, loss='mse')
        print self.model.summary()
