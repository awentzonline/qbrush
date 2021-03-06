from keras import backend as K
from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D,
    Input, LeakyReLU, merge, RepeatVector
)
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from qbrush.advantage import AdvantageAggregator
from qbrush.agent import QAgent
from qbrush.layers import BroadcastDim, ImageNetMean


class EtchASketchAgent(QAgent):
    def build_model(self):
        kernel_size = 5
        # input: position, canvas, target
        p = position_in = Input(shape=self.position_shape)
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([position_in, canvas_in, target_in], mode='concat')

        base_filters = 32
        x = Convolution2D(
            base_filters, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 2, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 4, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        y = x = Flatten()(x)
        # combine
        #y = merge([p, x], mode='concat')
        y = Dense(base_filters * 8)(y)
        y = Dropout(0.3)(y)
        y = LeakyReLU()(y)
        y = Dense(self.num_actions)(y)
        model = Model([position_in, canvas_in, target_in], y)
        optimizer = RMSprop(lr=0.0000625, rho=0.99, clipnorm=10.)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    @property
    def position_shape(self):
        if K.image_dim_ordering() == 'tf':
            return self.canvas_shape[:-1] + (1,)
        else:
            return (1,) + self.canvas_shape[1:]


class EtchASketchAdvantageAgent(EtchASketchAgent):
    def build_model(self):
        kernel_size = 5
        # input: position, canvas, target
        p = position_in = Input(shape=self.position_shape)
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([position_in, canvas_in, target_in], mode='concat')
        base_filters = 32
        x = Convolution2D(
            base_filters, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 2, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 4, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        # dueling advantage learning
        v = Dense(base_filters * 8)(x)
        v = Dropout(0.3)(v)
        v = LeakyReLU()(v)
        v = Dense(1)(v)

        a = Dense(base_filters * 8)(x)
        a = Dropout(0.3)(a)
        a = LeakyReLU()(a)
        a = Dense(self.num_actions)(a)

        q = AdvantageAggregator()([v, a])

        model = Model([position_in, canvas_in, target_in], q)
        optimizer = RMSprop(lr=0.0000625, rho=0.99, clipnorm=10.)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def model_custom_objects(self, **kwargs):
        return super(EtchASketchAdvantageAgent, self).model_custom_objects(
            AdvantageAggregator=AdvantageAggregator,
            ImageNetMean=ImageNetMean,
            BroadcastDim=BroadcastDim,
            **kwargs
        )


class EtchASketchFCAdvantageAgent(EtchASketchAdvantageAgent):
    '''Fully-convolutional advantage agent'''
    def build_model(self):
        kernel_size = 5
        # input: position, canvas, target
        p = position_in = Input(shape=self.position_shape)
        canvas_in = Input(shape=self.canvas_shape)
        target_in = Input(shape=self.canvas_shape)
        merged_inputs = merge([
            position_in,
            ImageNetMean()(canvas_in),
            ImageNetMean()(target_in)
        ], mode='concat')
        base_filters = 32
        x = Convolution2D(
            base_filters, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(merged_inputs)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 2, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        x = Convolution2D(
            base_filters * 4, kernel_size, kernel_size, subsample=(2, 2), border_mode='same')(x)
        x = LeakyReLU()(x)
        fc_kernel_size = kernel_size
        # dueling advantage learning
        v = Convolution2D(base_filters * 8, fc_kernel_size, fc_kernel_size, border_mode='same')(x)
        v = LeakyReLU()(v)
        v = Convolution2D(1, 1, 1, border_mode='same')(v)
        v = GlobalAveragePooling2D()(v)

        a = Convolution2D(base_filters * 8, fc_kernel_size, fc_kernel_size, border_mode='same')(x)
        a = LeakyReLU()(a)
        a = Convolution2D(self.num_actions, 1, 1, border_mode='same')(a)
        a = GlobalAveragePooling2D()(a)

        q = AdvantageAggregator()([v, a])

        model = Model([position_in, canvas_in, target_in], q)
        optimizer = RMSprop(lr=0.0000625, rho=0.99, clipnorm=10.)
        model.compile(optimizer=optimizer, loss='mse')
        return model
