import torch
import random
import os
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
RECOED_FILE = './record.txt'


class Agent:

    def __init__(self, model_file_path='./model/model.pth'):
        self.n_games = 0
        self.epsilon = 0  # randomrate
        self.gama = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_Qnet(768, 512, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gama)
        
        if os.path.exists(model_file_path):
            self.model.load_state_dict(torch.load(
                model_file_path))  # .\model\model.pth
            self.model.eval()

    def get_state(self, game):
        state = []
        for i in range(0, 32):
            temp = []
            for j in range(0, 24):
                val = 0
                p = Point(i*20, j*20)
                if p in game.snake:
                    val = 1
                elif p in game.food:
                    val = 2
                temp.append(val)
            state.append(temp)

        return state

    def get_state2(self, game):
        # (640/20)x(480/20)
        head = game.snake[0]
        point_r = Point(head.x + 20, head.y)
        point_l = Point(head.x - 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Dander Straight
            dir_r and game.is_collision(point_r) or
            dir_l and game.is_collision(point_l) or
            dir_u and game.is_collision(point_u) or
            dir_d and game.is_collision(point_d),

            # Danger right
            dir_u and game.is_collision(point_r) or
            dir_d and game.is_collision(point_l) or
            dir_l and game.is_collision(point_u) or
            dir_r and game.is_collision(point_d),

            # Danger left
            dir_d and game.is_collision(point_r) or
            dir_u and game.is_collision(point_l) or
            dir_r and game.is_collision(point_u) or
            dir_l and game.is_collision(point_d),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y,  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if(random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            state0 = torch.unsqueeze(state0, 0)
            prediction = self.model(state0)
            move = torch.argmax(prediction[0]).item()
            final_move[move] = 1

        return final_move


def load_record():
    if os.path.exists(RECOED_FILE):
        with open(RECOED_FILE, mode='r', encoding='utf-8') as file:
            return file.read()
    else:
        return '0'


def save_record(new_record):
    with open(RECOED_FILE, mode='w', encoding='utf-8') as file:
        file.write(str(new_record))


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = int(load_record())
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        curr_state = agent.get_state(game)

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
                save_record(record)
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            # TODO: plot
            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
