import argparse
import os

import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image

from qbrush import image_preprocessors
from qbrush.etch_a_sketch.agent import (
    EtchASketchAgent, EtchASketchAdvantageAgent, EtchASketchFCAdvantageAgent)
from qbrush.etch_a_sketch.environment import (
    EASEnvironment, EASFlawlessRunEnvironment, EASSingleLifetimeRewardEnvironment)
from qbrush.image_dataset import ImageDataset
from qbrush.trainer import Trainer


if __name__ == '__main__':
    agent_class =  EtchASketchAdvantageAgent  #EtchASketchAgent  #EtchASketchFCAdvantageAgent  #
    environment_class = EASFlawlessRunEnvironment
    # get config
    arg_parser = argparse.ArgumentParser('QBrush')
    environment_class.add_to_arg_parser(arg_parser)
    arg_parser.add_argument('target_glob')
    arg_parser.add_argument('--discount', type=float, default=0.95)
    arg_parser.add_argument('--episodes', type=int, default=1)
    arg_parser.add_argument('--epsilon', type=float, default=1.0)
    arg_parser.add_argument('--min-epsilon', type=float, default=0.05)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--sim-steps', type=int, default=250)
    arg_parser.add_argument('--learn-steps', type=int, default=500)
    arg_parser.add_argument('--num-canvases', type=int, default=3)
    arg_parser.add_argument('--output-path', type=str, default='./output')
    arg_parser.add_argument('--ignore-existing', action='store_true')
    arg_parser.add_argument('--model-name', default='eas_agent')
    arg_parser.add_argument('--save-rate', type=int, default=10)
    config = arg_parser.parse_args()


    print('loading images from {}'.format(config.target_glob))
    image_dataset = ImageDataset(config.target_glob, preprocessors=[
        image_preprocessors.greyscale,
        #image_preprocessors.rgb,
        image_preprocessors.resize((config.width, config.height))
    ])
    image_dataset.save_grid(
        os.path.join(config.output_path, 'dataset_sample.png')
    )
    config.channels = image_dataset.num_channels
    #target_image.save('grey_input.jpg')
    print('creating environment')
    environment = environment_class(config, num_actors=config.num_canvases)
    print('creating agent')
    agent = agent_class(
        environment.image_shape, environment.num_actions, discount=config.discount,
        ignore_existing=config.ignore_existing, name=config.model_name
    )
    print('simulating...')
    epsilon = config.epsilon
    d_epsilon = 1. / (config.episodes * config.epochs) * config.epsilon
    trainer = Trainer(config, agent, environment)
    for epoch_i in range(config.epochs):
        print('epoch {}'.format(epoch_i))
        environment.is_training = True
        for episode_i in range(config.episodes):
            print('episode {}.{} / epsilon = {}'.format(epoch_i, episode_i, epsilon))
            environment.update_targets(
                image_dataset.get_batch(config.num_canvases)
            )
            history, rewards = trainer.train(
                epsilon=epsilon, train_p=1.0, max_steps=config.learn_steps
            )
            if history:
                print('Loss: min: {} mean: {} max: {}'.format(
                    np.min(history), np.mean(history), np.max(history)
                ))
            if rewards:
                print('Rewards: min: {} mean: {} max: {}'.format(
                    np.min(rewards), np.mean(rewards), np.max(rewards)
                ))
            if (episode_i + 1) % config.save_rate == 0:
                agent.save_model()
            epsilon = max(config.min_epsilon, epsilon - d_epsilon)
        environment.is_training = False
        trainer.train(
            epsilon=config.min_epsilon, train_p=0.,
            max_steps=config.sim_steps
        )
        environment.save_canvas_state('epoch_{}.png'.format(epoch_i))
        environment.reset()
