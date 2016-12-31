import argparse
import os

import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image

from qbrush import image_preprocessors
from qbrush.etch_a_sketch.agent import EtchASketchAgent
from qbrush.etch_a_sketch.environment import EtchASketchEnvironment
from qbrush.image_dataset import ImageDataset


if __name__ == '__main__':
    agent_class = EtchASketchAgent
    environment_class = EtchASketchEnvironment
    # get config
    arg_parser = argparse.ArgumentParser('QBrush')
    environment_class.add_to_arg_parser(arg_parser)
    arg_parser.add_argument('target_image')
    arg_parser.add_argument('--discount', type=float, default=0.9)
    arg_parser.add_argument('--episodes', type=int, default=5)
    arg_parser.add_argument('--epsilon', type=float, default=1.0)
    arg_parser.add_argument('--min-epsilon', type=float, default=0.1)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--sim-steps', type=int, default=300)
    arg_parser.add_argument('--learn-steps', type=int, default=100)
    arg_parser.add_argument('--num-canvases', type=int, default=3)
    arg_parser.add_argument('--output-path', type=str, default='./output')
    config = arg_parser.parse_args()

    print('loading images from {}'.format(config.target_image))
    image_dataset = ImageDataset(config.target_image, preprocessors=[
        image_preprocessors.greyscale,
        image_preprocessors.resize((config.width, config.height))
    ])
    image_dataset.save_grid(
        os.path.join(config.output_path, 'dataset_sample.png')
    )
    #target_image.save('grey_input.jpg')
    print('creating environment')
    environment = environment_class(config, num_canvases=config.num_canvases)
    print('creating agent')
    agent = agent_class(
        environment.image_shape, environment.num_actions, discount=config.discount
    )
    print('simulating...')
    epsilon = config.epsilon
    d_epsilon = 1. / (config.episodes * config.epochs) * config.epsilon
    for epoch_i in range(config.epochs):
        print('epoch {}'.format(epoch_i))
        for episode_i in range(config.episodes):
            print('episode {}.{} / epsilon = {}'.format(epoch_i, episode_i, epsilon))
            environment.update_targets(
                image_dataset.get_batch(config.num_canvases)
            )
            history = environment.simulate(
                agent, epsilon=epsilon, train_p=0.5, max_steps=config.learn_steps
            )
            print('Loss: min: {} mean: {} max: {}'.format(
                np.min(history), np.mean(history), np.max(history)
            ))
            environment.reset()
            epsilon = max(config.min_epsilon, epsilon - d_epsilon)
        environment.simulate(
            agent, epsilon=config.min_epsilon, train_p=0.,
            max_steps=config.sim_steps
        )
        environment.save_image_state('epoch_{}.png'.format(epoch_i))
        environment.reset()
