import numpy as np
import pygame
import sys
from settings import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


class DefaultImediateReward:
    COLLISION_WALL = -10
    COLLISION_SELF = -10
    LOOP = -10
    SCORED = 10
    CLOSE_TO_FOOD = 0
    FAR_FROM_FOOD = 0
    MID_TO_FOOD = 0
    VERY_FAR_FROM_FOOD = 0
    EMPTY_CELL = 0
    DEFAULT_MOVING_CLOSER = 0
    MOVING_AWAY = 0


class Snake:

    def __init__(self, test=0):
        # 初始化 Pygame
        pygame.init()
        self.test = test
        self.TITLE = "g1"
        self.point = False
        self.is_close = False

        # 設定顏色
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)

        # 設定蛇的初始速度和大小
        self.snake_size = 20
        self.snake_speed = 10

        self.w = 200
        self.h = 200

        # 設定食物的初始位置和大小
        self.food_size = self.snake_size

        # 定義蛇的移動方向
        self.UP = 'UP'
        self.DOWN = 'DOWN'
        self.LEFT = 'LEFT'
        self.RIGHT = 'RIGHT'
        self.direction = self.RIGHT
        if test:
            self.action = [0, 1, 0, 0]
        self.reset()

        # 創建遊戲視窗
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(self.TITLE)

    def reset(self):
        self.direction = self.RIGHT
        self.play_num = 0

        # 定義食物的初始位置
        self.food_position = (random.randrange(1, (self.w // self.snake_size)) * self.snake_size,
                              random.randrange(1, (self.h // self.snake_size)) * self.snake_size)

        # 定義蛇的初始位置
        self.snake = [(self.w // 2, self.h // 2)]

        self.score = 0

    def chang_name(self, name):
        self.TITLE = name
        pygame.display.set_caption(self.TITLE)

    def draw_snake_and_food(self):
        # 畫出食物
        pygame.draw.rect(self.screen, self.red,
                         pygame.Rect(self.food_position[0], self.food_position[1], self.food_size, self.food_size))

        # 畫出蛇的頭部
        head_x, head_y = self.snake[0]
        pygame.draw.rect(self.screen, self.green, pygame.Rect(head_x, head_y, self.snake_size, self.snake_size))

        # 畫出蛇的身體
        for pos in self.snake[1:]:
            pygame.draw.rect(self.screen, self.white, pygame.Rect(pos[0], pos[1], self.snake_size, self.snake_size))
            pygame.draw.rect(self.screen, self.black, pygame.Rect(pos[0], pos[1], self.snake_size, self.snake_size), 2)

    def check_collision(self):
        head_x, head_y = self.snake[0]

        # 檢查是否碰到牆壁
        if head_x < 0 or head_x >= self.w or head_y < 0 or head_y >= self.h:
            return True

        # 檢查是否碰到自己的身體
        if (head_x, head_y) in self.snake[1:]:
            return True

        return False

    def control(self, action):
        if action[2] and self.direction != self.DOWN and self.direction != self.UP:
            self.direction = self.UP
        elif action[3] and self.direction != self.UP and self.direction != self.DOWN:
            self.direction = self.DOWN
        elif action[0] and self.direction != self.RIGHT and self.direction != self.LEFT:
            self.direction = self.LEFT
        elif action[1] and self.direction != self.LEFT and self.direction != self.RIGHT:
            self.direction = self.RIGHT

    def change_control(self, action):
        if self.direction == self.LEFT:
            if action[0]:
                return [1, 0, 0, 0]
            elif action[1]:
                return [0, 0, 1, 0]
            elif action[2]:
                return [0, 0, 0, 1]
        elif self.direction == self.RIGHT:
            if action[0]:
                return [0, 1, 0, 0]
            elif action[1]:
                return [0, 0, 0, 1]
            elif action[2]:
                return [0, 0, 1, 0]
        elif self.direction == self.UP:
            if action[0]:
                return [0, 0, 1, 0]
            elif action[1]:
                return [0, 1, 0, 0]
            elif action[2]:
                return [1, 0, 0, 0]
        elif self.direction == self.DOWN:
            if action[0]:
                return [0, 0, 0, 1]
            elif action[1]:
                return [1, 0, 0, 0]
            elif action[2]:
                return [0, 1, 0, 0]

    def move(self):
        # 移動蛇的尾部（從尾巴開始）
        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = (self.snake[i - 1][0], self.snake[i - 1][1])

        # 移動蛇的頭部
        if self.direction == self.UP:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] - self.snake_size)
        elif self.direction == self.DOWN:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] + self.snake_size)
        elif self.direction == self.LEFT:
            self.snake[0] = (self.snake[0][0] - self.snake_size, self.snake[0][1])
        elif self.direction == self.RIGHT:
            self.snake[0] = (self.snake[0][0] + self.snake_size, self.snake[0][1])

    def get_direction(self):
        if self.direction == self.LEFT:
            return [1, 0, 0, 0]
        if self.direction == self.RIGHT:
            return [0, 1, 0, 0]
        if self.direction == self.UP:
            return [0, 0, 1, 0]
        if self.direction == self.DOWN:
            return [0, 0, 0, 1]

    def get_point(self):
        # 在檢查是否吃到食物的部分
        if self.snake[0][0] == self.food_position[0] and self.snake[0][1] == self.food_position[1]:
            self.point = True
            while True:
                new_food_position = (random.randrange(1, (self.w // self.snake_size)) * self.snake_size,
                                     random.randrange(1, (self.h // self.snake_size)) * self.snake_size)

                # 檢查新生成的食物位置是否與蛇的身體重疊
                if new_food_position not in self.snake:
                    break

            self.food_position = new_food_position

            # 在蛇的尾部新增一個段落
            self.snake.append(self.snake[-1])

    def is_close_f(self):
        hx, hy = self.snake[0]
        fx, fy = self.food_position[0], self.food_position[1]
        if fx < hx and self.direction == self.LEFT:  # left
            self.is_close = True
        elif fx > hx and self.direction == self.RIGHT:  # right
            self.is_close = True
        elif fy < hy and self.direction == self.UP:  # up
            self.is_close = True
        elif fy > hy and self.direction == self.DOWN:  # down
            self.is_close = True

    def get_key(self):
        if self.test:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.action = [0, 0, 1, 0]
                    elif event.key == pygame.K_DOWN:
                        self.action = [0, 0, 0, 1]
                    elif event.key == pygame.K_LEFT:
                        self.action = [1, 0, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        self.action = [0, 1, 0, 0]
            return self.action

        else:
            action = [1, 0, 0]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = [1, 0, 0]
                    elif event.key == pygame.K_LEFT:
                        action = [0, 0, 1]
                    elif event.key == pygame.K_RIGHT:
                        action = [0, 1, 0]
            return action

    def play_step(self, action, kwargs=None):
        if not self.test:
            if kwargs is None:
                kwargs = {}

            self.play_num += 1
            if self.play_num < 3:
                self.snake.append(self.snake[-1])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            # 清空屏幕
            self.screen.fill(self.black)
            action = self.change_control(action)
            self.control(action)
            self.move()

            reward = kwargs.get('very_far_range', DefaultImediateReward.VERY_FAR_FROM_FOOD)

            terminal = False

            # 檢查是否撞到牆壁或自己
            if self.check_collision():
                terminal = True
                reward = kwargs.get('col_wall', DefaultImediateReward.COLLISION_WALL)
                reward = reward * (BAD_REWARD ** (len(self.snake) - 3))
                return reward, terminal, self.score

            if self.play_num > kwargs.get('kill_frame', DEFAULT_KILL_FRAME) * (len(self.snake) - 1):
                terminal = True
                reward = kwargs.get('loop', DefaultImediateReward.LOOP)
                reward = reward * (BAD_REWARD ** (len(self.snake) - 3))
                return reward, terminal, self.score

            self.get_point()

            self.is_close_f()

            if self.is_close:
                reward = kwargs.get('close_f', DefaultImediateReward.CLOSE_TO_FOOD)
                self.is_close = False
            elif not self.is_close:
                reward = kwargs.get('far_f', DefaultImediateReward.FAR_FROM_FOOD)

            if self.point:
                self.score += 1
                reward = kwargs.get('scored', DefaultImediateReward.SCORED)
                reward = reward * (GOOD_REWARD ** (len(self.snake) - 3))
                self.point = False

            # 繪製蛇和食物
            self.draw_snake_and_food()
            pygame.display.flip()

            return reward, terminal, self.score

        else:
            if kwargs is None:
                kwargs = {}

            self.play_num += 1
            if self.play_num < 3:
                self.snake.append(self.snake[-1])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            # 清空屏幕
            self.screen.fill(self.black)
            self.control(action)
            self.move()

            reward = kwargs.get('very_far_range', DefaultImediateReward.VERY_FAR_FROM_FOOD)
            terminal = False

            # 檢查是否撞到牆壁或自己
            if self.check_collision():
                terminal = True
                reward = kwargs.get('col_wall', DefaultImediateReward.COLLISION_WALL)
                reward = reward * (BAD_REWARD ** (len(self.snake) - 3))
                return reward, terminal, self.score

            if self.play_num > kwargs.get('kill_frame', DEFAULT_KILL_FRAME) * (len(self.snake) - 1):
                terminal = True
                reward = kwargs.get('loop', DefaultImediateReward.LOOP)
                reward = reward * (BAD_REWARD ** (len(self.snake) - 3))
                return reward, terminal, self.score

            self.get_point()

            self.is_close_f()

            if self.is_close:
                reward = kwargs.get('close_f', DefaultImediateReward.CLOSE_TO_FOOD)
                self.is_close = False
            elif not self.is_close:
                reward = kwargs.get('far_f', DefaultImediateReward.FAR_FROM_FOOD)

            if self.point:
                self.score += 1
                reward = kwargs.get('scored', DefaultImediateReward.SCORED)
                reward = reward * (GOOD_REWARD ** (len(self.snake) - 3))
                self.point = False

            # 繪製蛇和食物
            self.draw_snake_and_food()
            pygame.display.flip()

            return reward, terminal, self.score

    import numpy as np

    def cnn_input(self):
        cnn_input = np.zeros(((self.h // self.snake_size) + 1, (self.w // self.snake_size) + 1), dtype=int)
        snake_head = [self.snake[0][1] // self.snake_size, self.snake[0][0] // self.snake_size]
        snake_body = [
            [body[1] // self.snake_size, body[0] // self.snake_size]
            for body in self.snake[1:-1]
        ]
        snake_tail = [self.snake[-1][1] // self.snake_size, self.snake[-1][0] // self.snake_size]
        food = [self.food_position[1] // self.food_size, self.food_position[0] // self.food_size]

        cnn_input[tuple(snake_head)] = 1  # 蛇頭設為 1
        for body_pos in snake_body:  # 蛇身設為 2
            cnn_input[tuple(body_pos)] = 2
        cnn_input[tuple(snake_tail)] = 3  # 蛇尾設為 3
        cnn_input[tuple(food)] = 4  # 食物設為 4

        return cnn_input


if __name__ == "__main__":
    game = Snake(test=1)
    clock = pygame.time.Clock()
    while True:
        action = game.get_key()
        reward, terminal, score = game.play_step(action)
        print(game.cnn_input())
        clock.tick(1)  # 控制遊戲速度
        if terminal:
            print(score)
            break
