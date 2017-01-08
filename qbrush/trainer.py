import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, config, agent, environment):
        self.config = config
        self.agent = agent
        self.environment = environment

    def train(self, max_steps=1000, epsilon=0.5, train_p=0.0):
        last_state = None
        history = []
        rewards = []
        self.environment.reset()
        for step_i in tqdm(range(max_steps)):
            terminal = step_i == max_steps - 1
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(
                    0, self.environment.num_actions, (self.environment.num_actors,)
                )
            else:
                action = self.agent.policy(self.environment.get_state())

            this_state, reward, terminal, info = self.environment.step(action)
            rewards.append(reward)

            if last_state and train_p and (terminal.any() or reward.any() >= 0. or np.random.uniform(0., 1.) < train_p):
                loss = self.agent.train_step(last_state, action, reward, this_state, terminal)
                self.agent.train_target_model()
                history.append(loss)

            last_state = this_state
            if np.random.uniform(0., 1.) < 0.1:
                self.environment.save_canvas_state('output.png')
        return history, rewards
