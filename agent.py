from ast import main
from configparser import NoSectionError
import imp
from mimetypes import init
from random import random
import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomrate
        self.gama = 0  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # TODO: model, trainer

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        curr_state = agent.get_state()

        # get move
        final_move = agent.get_action(curr_state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            curr_state, final_move, reward, new_state, done)

        # remember
        agent.remember(curr_state, final_move, reward, new_state, done)

        if done:
            # train the long memory and plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score

                # agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            # TODO: plot

if __name__ == "__main__":
    train()
