import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, config):
        self.config = config

    def train(self, agent, environment, max_steps=1000, epsilon=0.5, train_p=0.0):
        last_state = None
        history = []
        environment.reset()
        for step_i in tqdm(range(max_steps)):
            terminal = step_i == max_steps - 1
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(
                    0, environment.num_actions, (environment.num_actors,)
                )
            else:
                action = agent.policy(environment.get_state())

            this_state, reward, is_finished, info = environment.step(action)

            if last_state and train_p and (reward.any() >= 0. or np.random.uniform(0., 1.) < train_p):
                loss = agent.train_step(last_state, action, reward, this_state)
                history.append(loss)

            last_state = this_state
            if np.random.uniform(0., 1.) < 0.1:
                environment.save_canvas_state('output.png')
        return history
