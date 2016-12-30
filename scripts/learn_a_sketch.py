import argparse
import os

from keras.preprocessing.image import img_to_array
from PIL import Image

from qbrush.etch_a_sketch.agent import EtchASketchAgent
from qbrush.etch_a_sketch.environment import EtchASketchEnvironment


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
    arg_parser.add_argument('--output-path', type=str, default='./output')
    config = arg_parser.parse_args()

    # load target image
    target_image = Image.open(config.target_image)
    target_image = target_image.resize((config.width, config.height))
    target_image = target_image.convert('L').convert('RGB')
    target_image_arr = img_to_array(target_image)
    #target_image.save('grey_input.jpg')
    print('creating environment')
    environment = environment_class(target_image_arr, config)
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
            environment.simulate(agent, epsilon=epsilon, train_p=0.5)
            environment.reset()
            epsilon = max(config.min_epsilon, epsilon - d_epsilon)
        environment.simulate(agent, epsilon=config.min_epsilon, train_p=0.)
        environment.save_image_state('epoch_{}.png'.format(epoch_i))
        environment.reset()
