from .build_model import build_model
import numpy as np
from collections import deque
import random

class QLearningAgent:
    def __init__(self, state_shape, num_actions):
        self.network = build_model(state_shape, num_actions)
        self.state_shape = state_shape
        self.num_actions = num_actions
        np.random.seed(0)
        self.epsilon = 0.1
        self.gamma = 0.9
        self.batch_size = 20
        self.experience_replay = deque(maxlen=2000)
        self.possible_actions = np.eye(num_actions)

    def act(self, observation, epsilon=0.0):
        if np.random.random() < epsilon:
            return self.possible_actions[np.random.choice(self.num_actions)]
        reshaped_observation = np.expand_dims(observation, axis=0)
        results = self.network.predict(reshaped_observation)
        return self.possible_actions[np.argmax(results)]

    def add_experience(self, state, action, reward, next_state, terminal):
        self.experience_replay.append((state, action, reward, next_state, terminal))

    def sample_from_experience(self, batch_size):
        if len(self.experience_replay) < batch_size:
            batch_size = len(self.experience_replay)
        samples = random.sample(self.experience_replay, batch_size)
        states = np.concatenate([np.expand_dims(s[0], axis=0) for s in samples], axis=0)
        actions = np.concatenate([np.expand_dims(s[1], axis=0) for s in samples], axis=0)
        rewards = np.array([s[2] for s in samples])
        next_states = np.concatenate([np.expand_dims(s[3], axis=0) for s in samples], axis=0)
        terminals = np.array([s[4] for s in samples])
        return states, actions, rewards, next_states, terminals

    def q_learning_step(self, state, action, reward, next_state, terminal, batch_size):
        self.add_experience(state, action, reward, next_state, terminal)
        if batch_size > len(self.experience_replay): return
        states, actions, rewards, next_states, terminals = self.sample_from_experience(batch_size)
        targets = self.network.predict(states)
        action_mask = np.array(actions, dtype=bool)
        next_targets = self.network.predict(next_states)
        targets[action_mask] = rewards
        targets[action_mask] += self.gamma*np.amax(next_targets, axis=1)*np.invert(terminals)
        self.network.fit(states, targets, epochs=1, verbose=0)

