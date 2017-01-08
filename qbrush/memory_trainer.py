import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, config, agent, environment, memory):
        self.config = config
        self.agent = agent
        self.environment = environment
        self.memory = memory

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

            if last_state:
                self.memory.add((last_state, action, reward, this_state, terminal))
            if self.memory.size > 100 and np.random.uniform(0., 1.) < train_p:
                losses = self.train_from_memory()
                self.agent.train_target_model()
                history += losses

            last_state = this_state
            if np.random.uniform(0., 1.) < 0.1:
                self.environment.save_canvas_state('output.png')
        return history, rewards

    def train_from_memory(self, num_batches=1):
        full_history = []
        for i in range(num_batches):
            for s, a, r, s2, t in self.memory.sample(num_batches):
                full_history.append(self.agent.train_step(s, a, r, s2, t))
        return full_history
