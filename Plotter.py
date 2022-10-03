import matplotlib.pyplot as plt
from IPython import display

plt.ion()

class Plotter:

    def __init__(self, trailing_win):
        self.window                     = trailing_win
        
        self.scores                     = []
        self.mean_scores                = []
        self.trailing_mean_scores       = []
        
        self.eff_scores                 = []
        self.mean_eff_scores            = []
        self.trailing_mean_eff_scores   = []


    def appender(self, score, n_steps):
        win = min(self.window, max(1, len(self.scores)))

        # Score
        self.scores.append(score)
        
        # Mean score
        mean_score = sum(self.scores) / (len(self.scores)) # Minus 1 due to scores being initialize from zero
        self.mean_scores.append(mean_score)

        # Trailing average score
        trailing_mean_score = sum(self.scores[-win:]) / win
        self.trailing_mean_scores.append(trailing_mean_score)


        # Efficiency score
        eff_score = score / n_steps
        self.eff_scores.append(eff_score)

        # Average efficiency scores
        mean_eff_score = sum(self.eff_scores) / (len(self.scores))
        self.mean_eff_scores.append(mean_eff_score) 

        # Trailing average efficiency score
        trailing_mean_eff_score = sum(self.eff_scores[-win:]) / win
        self.trailing_mean_eff_scores.append(trailing_mean_eff_score)


    def plot(self):

        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(1).clear()
        plt.figure(2).clear()

        scores                  = self.scores
        mean_scores             = self.mean_scores
        trailing_mean_scores    = self.trailing_mean_scores

        eff_scores              = self.eff_scores
        mean_eff_scores         = self.mean_eff_scores
        trailing_mean_eff_scores= self.trailing_mean_eff_scores

        plt.figure(1)
        plt.title('Training performance')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label = 'Current')
        plt.plot(mean_scores, label = 'Avg.')
        plt.plot(trailing_mean_scores, label=f'Trailing {self.window}-game avg.')
        plt.ylim(ymin=0)    
        plt.text(len(scores)-1, scores[-1], str(round(scores[-1],2)))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1],2)))
        plt.text(len(trailing_mean_scores)-1, trailing_mean_scores[-1], str(round(trailing_mean_scores[-1],2)))
        plt.legend(loc="upper left")

        plt.figure(2)
        plt.title('Training efficiency')
        plt.xlabel('Number of Games')
        plt.ylabel('Score per game steps')
        plt.plot(eff_scores, label = 'Current')
        plt.plot(mean_eff_scores, label = 'Avg.')
        plt.plot(trailing_mean_eff_scores, label=f'Trailing {self.window}-game avg.')
        plt.ylim(ymin=0)
        plt.text(len(eff_scores)-1, eff_scores[-1], str(round(eff_scores[-1],3)))
        plt.text(len(mean_eff_scores)-1, mean_eff_scores[-1], str(round(mean_eff_scores[-1],3)))
        plt.text(len(trailing_mean_eff_scores)-1, trailing_mean_eff_scores[-1], str(round(trailing_mean_eff_scores[-1],3)))
        plt.legend(loc="upper left")

