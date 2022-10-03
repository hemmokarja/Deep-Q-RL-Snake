import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random



class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden_layers, out_size=3):
        super().__init__()
        self.linear_in  = nn.Linear(input_size, hidden_size)
        self.relu_in = nn.ReLU()
        self.linear_rest = self.hidden_block(n_hidden_layers, hidden_size, out_size)

    def forward(self, X):
        out = self.linear_in(X)
        out = self.relu_in(out)
        out = self.linear_rest(out)
        return out

    def hidden_block(self, n_hidden_layers, hidden_size, out_size=3):
        layers = []
        hidden_layer_set = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        ]
        # Add additional hidden layers
        for _ in range(n_hidden_layers-1):
            layers.extend(hidden_layer_set)

        # Add output layer
        layers.append(nn.Linear(hidden_size, out_size))
        
        return nn.Sequential(*layers)



class Trainer:
    
    def __init__(self, policy_net, target_net, lr, gamma, large_batch_size, small_batch_size, max_replay_buffer):
        
        self.policy_net = policy_net
        self.target_net = target_net

        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
        self.large_batch_size = large_batch_size
        self.small_batch_size = small_batch_size

        self.replay_buffer = deque(maxlen=max_replay_buffer)
        

    def train_step(self, state, action, reward, next_state, done):
        '''
        Training the model.
        
        First, recall that Qs represent the future discounted rewards until the end of the game.
        Thus, by construction, Q(current_state) = reward(current_state) + Q(next_state).
        If the game ends, then Q(current_state) = reward(current_state).

        The basis for training is the notion that a good network should be able to model the future 
        discounted rewards resulting from the taken action.

        The core idea for training a Deep Q Network is to:
        (1) Predict Q(current_state) with the policy net
        (2) Obtain reward(current_state), which represents the future reward from the actual action taken from the current state
        (3) Check if game has ended
            > if it has, then the target Q is reward
            > if not, then get all predicted Q values based on the next state *, and the target Q is the highest value
        (4) Make copy of the Q(current state), and replace the value which corresponds to the taken action's 
            index (0,1,2) with the new target Q
        (5) Train the neural net so that vector Q(current state) equals the same vector with the updated target Q
            at the index of the actual taken action. 

        * Notice that the next state's Q predictions are made using the target net. Target net may be updated at longer 
          intervals than the policy net, providing a more stable target prediction. The policy net is often easier to train
          if the way the target is predicted does not change each time the model is trained.
        '''

        # Get predicted Q values based on the current state
        current_qs = self.policy_net(state)

        # Current Qs provide the basis for the target vectors
        # his vector will be modified next to have the target Q at the index of the action
        target_qs = current_qs.clone()

        # Loop over the samples in a batch
        for idx in range(len(done)): 
            target_q = reward[idx] # Reward associated with the sample
            if not done[idx]:
                # Target Q is the sum of all discounted future rewards from the taken action
                next_state_qs = self.target_net(next_state[idx])
                target_q = reward[idx] + self.gamma * torch.max(next_state_qs)

            # Update target Qs
            # 'target_qs[idx]' is a row vector with Qs predicted from current state
            # Replace the value which corresponds to the taken action's index (0,1,2) with the updated, new Q
            # The new Q represents the discounted future rewards until the end of the game
            # A good model should be able to model the future discounted rewards resulting from the taken action, thus we use this vector as the target
            move_idx = torch.argmax(action).item()
            target_qs[idx][move_idx] = target_q

        # Perform training
        self.optimizer.zero_grad()              
        loss = self.loss(target_qs, current_qs) 
        loss.backward()
        self.optimizer.step()


    def memorize(self, transition):
        self.replay_buffer.append(transition) # Transition contains current_state, action, reward, next_state, done in a tuple


    def _sample_batch(self, batch_size):
        if len(self.replay_buffer) > batch_size:
            batch = random.sample(self.replay_buffer, batch_size)
        else:
            batch = self.replay_buffer
        return batch


    def train_large_batch(self):
        batch = self._sample_batch(self.large_batch_size) # Sample a batch of transitions
        batch_tensors = self.tensorize_batch(batch) # Turn transitions into tensors
        self.train_step(*batch_tensors) # Train model


    def train_small_batch(self):
        batch = self._sample_batch(self.small_batch_size) 
        batch_tensors = self.tensorize_batch(batch)
        self.train_step(*batch_tensors)


    @staticmethod
    def tensorize_batch(batch):           
        
        # At present, a single sample in batch is the entire transition (state, action, reward, next_state, done).
        # We want to separate all element types into their own tuples so that 
        # all states are in a single tuple, all actions in a single tuple, and so on.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Turn into tensors                                                     # Each sample is:
        states       = torch.tensor(np.array(states), dtype=torch.float)        # 11-dimensional binary vector
        next_states  = torch.tensor(np.array(next_states), dtype=torch.float)   # 11-dimensional binary vector
        actions      = torch.tensor(actions, dtype=torch.long)                  # 3-dimensional vector of zeros and a single "1" representing the take action which lead from state to next state
        rewards      = torch.tensor(rewards, dtype=torch.float)                 # Scalar

        # Check if the batch contains a single sample -> add num_samples dimension
        if len(states.shape) == 1:
            states       = states.unsqueeze(0)
            next_states  = next_states.unsqueeze(0)
            actions      = actions.unsqueeze(0)
            rewards      = rewards.unsqueeze(0)
            dones        = (dones, )                                            # Boolean

        return states, actions, rewards, next_states, dones