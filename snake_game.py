import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
GREEN = (0, 200, 50)
RED1 = (200, 0, 0)
RED2 = (200, 100, 100)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = []
        self._place_food(10)
        self.frame_iteration = 0

    def _place_food(self, num=1):
        if len(self.food) >= num:
            return

        i = 0
        while(i <= num):
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            p = Point(x, y)
            if p in self.snake or p in self.food:
                continue
            else:
                self.food.append(p)
                i += 1
            # self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward -= 2 * len(self.snake)
            return reward, game_over, self.score

        if self.frame_iteration > 100*len(self.snake):
            reward -= 100 // len(self.snake)

        # 4. place new food or just move
        if self.head in self.food:
            self.frame_iteration = 0
            self.score += 1
            reward += 2 * len(self.snake)
            self.food.remove(self.head)
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point == None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for i in range(len(self.snake)):
            pt = self.snake[i]
            if i == 0:
                pygame.draw.rect(self.display, RED1, pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, RED2,
                                 pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2,
                                 pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        for pt in self.food:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        x = self.head.x
        y = self.head.y

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        curr_dir_indx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # STRAIGHT : no change
            new_dir = clock_wise[curr_dir_indx]
        elif np.array_equal(action, [0, 1, 0]):
            # Right Turn : r -> d -> -> l -> u
            next_dir_indx = (curr_dir_indx + 1) % 4
            new_dir = clock_wise[next_dir_indx]
        elif np.array_equal(action, [0, 0, 1]):
            # Left Turn : r -> u -> -> l -> d
            next_dir_indx = (curr_dir_indx - 1) % 4
            new_dir = clock_wise[next_dir_indx]

        self.direction = new_dir

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
