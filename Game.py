import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

BLOCK_SIZE = 20 # How large blocks in snake are
SPEED = 100 # How fast frame updates



Point = namedtuple('Point', 'x, y')


class Direction(Enum):
    RIGHT   = 1
    LEFT    = 2
    UP      = 3
    DOWN    = 4


class SnakeGame:

    def __init__(self, w=640, h=480, ui='nokia'):
        self.w = w
        self.h = h

        if ui == 'nokia':
            self.FONT_COL            = (0,0,0)
            self.BACKGROUD_COL       = (150,190,90) # Greenish
            self.SNAKE_OUTLINE_COL   = (0,0,0)
            self.SNAKE_FILL_COL      = (45,45,45)
            self.FOOD_COL            = (0,0,0)

        else:
            self.FONT_COL            = (255,255,255)
            self.BACKGROUD_COL       = (0,0,0)
            self.SNAKE_OUTLINE_COL   = (0,0,255)
            self.SNAKE_FILL_COL      = (0,100,255)
            self.FOOD_COL            = (200,0,0) 


        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Init game state
        self.reset()


    def reset(self):
        # Init game state
        self.direction  = Direction.RIGHT
        self.head       = Point(self.w/2, self.h/2)
        self.snake      = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y), 
            Point(self.head.x - (2*BLOCK_SIZE), self.head.y)
            ]
        self.score              = 0
        self.frame_iteration    = 0
        self.food               = None
        self._place_food()


    def _place_food(self):
        x  = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y  = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # Check that food is not placed where the snake is currently located
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        
        self.frame_iteration += 1
        reward = 0
        done = False
        
        # 1: Collect user input (only quitting game as snake is AI controlled)
        for event in pygame.event.get(): # Get all user events that happened within one play_step            
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2: Move snake
        self._move(action)              # Update the head location
        self.snake.insert(0, self.head) # Not .append() because we want head at the beginning, not end

        # 3: Check if game over (either collision or stuck in a loop)
        if self.is_collision(): #or self.frame_iteration > 100*len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score, self.frame_iteration

        # 4: Else, place new food or just move
        if self.head == self.food: # Food eaten -> snake gets longer 
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop() # Food not eaten -> remove one block from tail (head moved to new location so this keeps the snake at same length)

        # 5: Update pygame UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6: Return game over and score
        done = False

        return reward, done, self.score, self.frame_iteration


    def _get_direction(self, action):
        '''Get direction based on action. Action is one hot encoded binary list [straight, right, left]'''

        clock_wise  = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx         = clock_wise.index(self.direction) # Get index of current direction

        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]       # Keep current direction

        elif np.array_equal(action, [0,1,0]):
            new_idx = (idx + 1) % 4         # Index 4 (does not exist) loops back to index zero
            new_dir = clock_wise[new_idx]   # Turn right
            
        else: # Has to be [0,0,1]
            new_idx = (idx - 1) % 4         # Index -1 (does not exist) loops back to index three
            new_dir = clock_wise[new_idx]   # Turn left

        self.direction = new_dir


    def _move(self, action):
        '''Updates head location.'''

        # Update direction
        self._get_direction(action)

        # Update head location
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x,y)


    def is_collision(self, point=None):

        if point is None:
            point = self.head

        # Boundary hit
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True

        # Snake tail hit (snake head the first element - does not need to be checked)
        if point in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(self.BACKGROUD_COL)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, self.SNAKE_OUTLINE_COL, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, self.SNAKE_FILL_COL, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, self.FOOD_COL, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = font.render('Score: ' + str(self.score), True, self.FONT_COL)
        self.display.blit(text, [0,0])
        
        # Render the changes on screen
        pygame.display.flip()
