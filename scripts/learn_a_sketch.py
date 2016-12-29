import argparse

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
    arg_parser.add_argument('--discount', default=0.9)
    arg_parser.add_argument('--episodes', default=10000)
    arg_parser.add_argument('--epsilon', default=1.0)
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
    d_epsilon = 1. / config.episodes * config.epsilon
    for episode_i in range(config.episodes):
        print('episode {} / epsilon {}'.format(episode_i, epsilon))
        environment.simulate(agent, epsilon=epsilon)
        environment.save_image_state('canvas.png')
        environment.reset()
        epsilon = max(0., epsilon - d_epsilon)
