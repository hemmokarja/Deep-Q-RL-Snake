
import torch
import numpy as np
import random
from Game import Point, Direction


class Agent:

    def __init__(self, strategy, state_stack_len, danger_dist=20):
        self.n_games    = 0
        self.n_steps    = 0
        self.strategy   = strategy
        self.state_stack_len = state_stack_len * 11 # State stack len measured in elements instead of states
        self.state_stack = np.zeros((self.state_stack_len, )) 
        self.danger_dist = danger_dist


    def get_state(self, game, done):
        if done:
            stack               = np.zeros((self.state_stack_len, ))
            self.state_stack    = stack
        else:

            # Get danger box around snake head
            head    = game.snake[0]
            point_l = Point(head.x - self.danger_dist, head.y)
            point_r = Point(head.x + self.danger_dist, head.y)
            point_u = Point(head.x, head.y - self.danger_dist)
            point_d = Point(head.x, head.y + self.danger_dist)

            # Get current direction (only one is True, others False)
            dir_l   = game.direction == Direction.LEFT
            dir_r   = game.direction == Direction.RIGHT
            dir_u   = game.direction == Direction.UP
            dir_d   = game.direction == Direction.DOWN

            # Get state 
            # Consists of 11 T/F variables
            state = [ 

                # VAR 1: Danger straight ahead
                (dir_l and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_r)) or
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)),

                # VAR 2: Danger on the right
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d)) or
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)),

                # VAR 3: Danger on the left
                (dir_l and game.is_collision(point_d)) or
                (dir_r and game.is_collision(point_u)) or
                (dir_u and game.is_collision(point_l)) or
                (dir_d and game.is_collision(point_r)),
                
                # VAR: 4-7 Direction (only one is True)
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                
                # VAR 8-11: Food location
                game.food.x < game.head.x,   # Food is on the left
                game.food.x > game.head.x,   # Food is on the right
                game.food.y < game.head.y,   # Food is above
                game.food.y > game.head.y,   # Food is below
            ]

            # Turn state into binary vector
            state = np.array(state, dtype=int)

            # Append it to current state stack and save the portion that is wanted
            self.state_stack = np.append(self.state_stack, state)[-(self.state_stack_len):]

        return self.state_stack


    def get_action(self, state, policy_net):
        '''Get an action based on state of game'''

        rate = self.strategy.get_exploration_rate(self.n_steps, self.n_games)
        self.n_steps += 1
        action = [0,0,0]

        # Sample random action
        if rate > random.random():
            move = random.randint(0,2)
            action[move] = 1
        # Predict the next best move using the policy net
        else:
            state = torch.tensor(state, dtype=torch.float)
            pred = policy_net(state)           
            move = torch.argmax(pred).item()
            action[move] = 1
        
        return action







