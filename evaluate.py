import tensorflow as tf
from tensorflow import keras

from dnn.agent import QLearningAgent

import os,sys

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

def done_callback(agent, world):
    for landmark in world.landmarks:
        if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) <= agent.size - landmark.size:
            return True
    return False


if __name__ == '__main__':
    # load scenario from script
    scenario = scenarios.load('simple_spread.py').Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=done_callback, shared_viewer = False)

    model = tf.keras.models.load_model('saved_model/latest')
    qagent = QLearningAgent(
            env.observation_space[0].shape, env.action_space[0].n)
    qagent.network = model

    num_epochs = 10
    for i in range(num_epochs):
        obs_n = env.reset()
        env.render()
        done_master = [False for _ in range(len(obs_n))]
        count = 0
        while not all(done_master) and count < 1000:
            count += 1
            act_n = []
            for j in range(len(obs_n)):
                act_n.append(qagent.act(obs_n[j], 0.0))
            obs_n, reward_n, done_n, _ = env.step(act_n)
            for j in range(len(obs_n)):
                if done_n[j]: done_master[j] = True
            env.render()

