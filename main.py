import random
from Plotter import Plotter
from Game import SnakeGame
from Model import DQN, Trainer
from Strategy import EpsilonGreedy, LinearDecay
from Agent import Agent

random.seed(1)

# Neural net constraints
n_hidden_layers     = 1
hidden_size         = 256
max_replay_buffer   = 100_000
large_batch_size    = 1_000
small_batch_size    = 1
learning_rate       = 0.001     
gamma               = 0.9       # Discount factor in the Bellman Equation
update_target_every = 1         # How often target net weights are updated

# Agent constraints
state_stack_len = 2             # Measured in number of states

# Plotter constants
trailing_win = 20

# Epsilon Greedy strategy constants
start_eps   = 1
end_eps     = 0.0
decay_eps   = 0.0006

# Linear Decay strategy constants
eps_constant = 100
base         = 200



# Main training loop
def train():

    # Istantiate Strategy 
    strategy = EpsilonGreedy(start_eps, end_eps, decay_eps)
    # strategy = LinearDecay(eps_constant, base)
    
    # Istantiate Agent
    agent = Agent(strategy, state_stack_len)

    # Istantiate Neural Nets
    policy_net = DQN(input_size=(11*state_stack_len), hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
    target_net = DQN(input_size=(11*state_stack_len), hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
    
    target_net.load_state_dict(policy_net.state_dict()) # Initialize target nets's weights with policy net's weights
    target_net.eval() # Set target net in predict-only mode (doesn't track gradients)

    # Instantiate Trainer
    trainer = Trainer(
        policy_net=policy_net, 
        target_net=target_net,
        lr=learning_rate, 
        gamma=gamma, 
        large_batch_size=large_batch_size, 
        small_batch_size=small_batch_size, 
        max_replay_buffer=max_replay_buffer
        )
    
    # Istantiate Snake Game
    game = SnakeGame(w=600, h=480, ui='nokia')
    
    # Istantiate Plotter
    plotter = Plotter(trailing_win)

    # Run game
    done = False
    while True:

        # 1: Get current state
        cur_state = agent.get_state(game, done)

        # 2: Get move based on current state
        action = agent.get_action(cur_state, policy_net)

        # 3: Perform action
        reward, done, score, n_steps = game.play_step(action)

        # 4: Get new state, corresponding to the taken action
        next_state = agent.get_state(game, done)

        # 5: Store transition in replay buffer
        trainer.memorize((cur_state, action, reward, next_state, done))

        # 6: Train with a small batch
        trainer.train_small_batch()

        # Check if game ended
        if done:
            game.reset()
            agent.n_games += 1
            done = False

            # Train with a large batch
            trainer.train_large_batch()

            # Update target net weights
            if agent.n_games % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Append scores & plot
            plotter.appender(score, n_steps)
            plotter.plot()
            

if __name__ == '__main__':
    train()


