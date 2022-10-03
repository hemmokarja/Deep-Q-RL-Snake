import math


class EpsilonGreedy:

    def __init__(self, start, end, decay):
        self.start  = start 
        self.end    = end
        self.decay  = decay

    def get_exploration_rate(self, n_steps, n_games):
        return self.end + (self.start - self.end) * math.exp(-1 * n_steps * self.decay)


class LinearDecay:

    def __init__(self, epsilon_constant, base):
        self.epsilon_constant   = epsilon_constant
        self.base               = base

    def get_exploration_rate(self, n_steps, n_games):
        rate = (self.epsilon_constant - n_games) / self.base
        return rate

