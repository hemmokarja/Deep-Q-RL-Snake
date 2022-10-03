# **Deep Q Reinforcement Learning Agent for playing Snake**

### **BACKGROUND**

**What is Reinforcement Learning**

Reinforcement learning (RL) is a machine learning method based on rewarding desired behaviors and/or punishing undesired ones. In general, a reinforcement learning agent is able to perceive and interpret its environment, take actions and learn through trial and error. This programs the agent to seek long-term and maximum overall reward to achieve an optimal solution. These long-term goals help prevent the agent from stalling on lesser goals. With time, the agent learns to avoid the negative and seek the positive. 

Breaking it down, the process of Reinforcement Learning involves these simple steps:
* Observation of the environment
* Deciding how to act using some strategy
* Acting accordingly
* Receiving a reward or penalty
* Learning from the experiences and refining our strategy
* Iterate until an optimal strategy is found

The following figure encapsulates the core logic behind Reinforcement learning in grpahical form:

![The-agent-environment-interaction-in-reinforcement-learning](https://user-images.githubusercontent.com/106996328/190202672-ebbbc5b1-c0bf-4e55-9667-bc4e6fb406db.png)


**What is Q-Learning?**
 
In order to understand Deep Q-Learning, one should first have a conception of Q-Learning. 

First off, it’s useful to know that the “Q” in Q-Learning stands for “quality”. Quality here represents how useful a given action is in gaining some future reward at any given state. Through trial and error, the Agent makes an effort to learn the Q-values associated with each state-action pair, which allows successful navigation of its environment.

How is the Q-value quantified, then? Q-value (Q(s,a)) is the expected value (cumulative discounted reward) of taking action a in state s and then following the optimal policy. More specifically, Q(s,a) s is computed using the Bellman Equation (https://en.wikipedia.org/wiki/Bellman_equation) 

![1*EQ-tDj-iMdsHlGKUR81Xgw](https://user-images.githubusercontent.com/106996328/190202001-57d920a5-d401-4b77-b59d-4ec43dfccab9.png)

How are the Q-values used in practice? The Agent will simply take the action associated with the highest Q-value in the current state. The caveat here is, though, that at the beginning, the Agent does not know the Q-values of any state; it can only learn them by trial and error, i.e., by exploring its environment. 

A central concept in Q-Learning is the Q-table. A Q-table is a simple data structure that is used to keep track of the states, actions, and their expected rewards. More specifically, the Q-table maps a state-action pair to a Q-value (the estimated optimal future sum of rewards) which the agent will learn. At the start of the Q-Learning algorithm, the Q-table is initialized to all zeros indicating that the agent doesn’t know anything about the world. As the agent tries out different actions at different states through trial and error, the agent learns each state-action pair’s expected reward and updates the Q-table with the new Q-value. Using trial and error to learn about the world is called Exploration. In contrast, explicitly choosing the best known action at a state is called Exploitation. Typically, at the beginning, the Agent is programmed to rely mostly on exploration, and then, once it has began to learn about its environment, it proggressively shifts more and more towards exploitation of the Q-table. An example of a Q-table below:

![Screenshot 2022-09-14 at 18 57 09](https://user-images.githubusercontent.com/106996328/190203971-f58800fc-13b9-45be-916b-45d7a1269119.png)

However, the obvious issue with Q-tables is the lack of scalability; they become practically infeasible in applications with vast numbers of possible states or actions (imagine constructing a Q-table for playing any console game with millions of possible states and hundreds and hundreds of possible actions). Deep Q-Learning was developed to address this problem, in particular.

**What is Deep Q-Learning?**

A core difference between Deep Q-Learning and “vanilla” Q-Learning is the implementation of the Q-table. Critically, Deep Q-Learning replaces the regular Q-table with a neural network. Rather than mapping a state-action pair to a Q-value, a neural network maps input states to (action, Q-value) pairs. In other words, the neural network used for estimating the best action (called, “policy net”) takes as input the current state, and gives as output the Q-values associated with each action. If we have three possible actions in the set, the network will output three Q-values, one associated with each action. The preferred action is then the one with the highest Q-value. 

While this part is simple enough, and easy to grasp, the intuition behind training the network may seem a bit less straightforward. In particular, exactly what should be the target vector (or ground truth) that is used for training? How do we know whether or not the policy net did a good job estimating the Q-values of each action? Recall that the Q-value is the discounted sum of all future rewards from taking action a in state s and then following the optimal policy. Thus, we can use as the target vector the estimated Q-values from the policy net, and replace the Q-value of the taken action with the discounted sum of all future rewards by following the optimal policy. The next question is what is this discounted sum of all future rewards? It can be computed using the Bellman Equation, with the expected future Q(s,a) in the equation being estimated by a neural network. A good model should be able to model the future discounted rewards resulting from the taken action, thus we use this vector as the target.

While the estimation of expected future Q(s,a) can be (somewhat paradoxically) carried out by the same policy net that is being trained, it’s more common to use a separate network called “target net”. The target net is identical in architecture to the policy net, but it is not trained independently at all; instead, the target net uses the policy net’s weights, which are updated only every Nth training iteration. This provides a more stable target for the policy net to learn as it is not “chasing its own tail”.

It goes wihtout saying, that providing a comprehensive review of Deep Q-Learning is way out of scope of this README file, but for those interested in the subject, I highly recommed the following lecture by Yannic Kilcher, in which he reviews the paper which first introduced Deep Q-Learning: https://www.youtube.com/watch?v=rFwQDDbYTm4

### **THIS PROJECT**

In this project, I implement a Deep Q-Learning model which learns to play Snake. There are three main elements in this project: the Game, the Model, and the Agent. Each of them are contained in their own files. The Game represents the environment. The environment is the Snake game modelled with PyGame and contains the functionality for playing the game. That is, it takes actions as inputs and updates the game and the screen accordingly, giving as output the reward at each frame iteration. The Model is the Deep Q-Network (DQN) used for estimating the Q-values. The file also includes the trainer class which contains the functionality for training the model. The Agent represents the player who requests states from the Game and by using them, requests actions from the Strategy. 

In addition, the list of files include Strategy, which contains two different strategies for choosing between exploitation and exploration; a Plotter, which is a helper module used for plotting the performance of the model; the main.py file, which is the main file used for running the program; and the requirements text file containing the dependencies.

As noted above, the main training file is used for running the program. The logic of the main running loop of the program is as follows:

**(1) Agent requests current state from the Game.** A single state state is a 11-dimensional binary vector. The first three boolean elements represent whether there is a danger straight ahead, on the right, or on the left in relation to the snake’s current direction (danger means a collision if moved one step further in that direction). The elements 4 to 7 represents the current direction of the snake, with only one being true, and the others false. The remaining elements from 8 to 11 represent the food location; whether it is located on the left, on the right, above, or below, in relation to the snake’s current direction and location. However, in this application, the received state is not a single state, but a so-called “state stack”, which is a concatenation of the last *n* (here, two) states. A state stack gives a richer representation of the current state than a single state.

**(2) Agent requests an action from the Strategy.** The action may be taken through exploration (higher likelihood in the beginning), or exploitation (higher likelihood later in the game). If the action is taken through exploration, it is selected randomly. In contrast, an action taken through exploitation is requested from the policy net. The received action is a 3-dimensional vector with two zeros and one one. The three elements represent whether the snake should turn left, right, or continue in the current direction.

**(3) Action is performed in the Game.** The action is fed to the game, and as output, we get the reward from the taken action (10 if food eaten, 0 if not, -10 if game ended), and a boolean variable representing whether the game ended. (We also get the current score and number of steps taken thus far, though, neither is critical for the algorithm).

**(4) Agent requests from the Game the next state that follows from the peformed action.** The received state is a similar 11-dimensional vector as in step (1). The next state is crucial for training the model. 

**(5) Memorize the transition to replay buffer.** Transition is the combination of the (i) current state from which the action is performed; (ii) the performed action; (iii) the reward resulting from the action; (iv) the next state to which the action led to; (v) the boolean representing whether the game ended as a result of the action. In other words, the transition represents entirely what happened in a particular game step. Replay buffer, in turn, is a container for storing these past transitions. Its use case is explained in the next section.

**(6) Train the policy net with a small batch.** In between the game steps, we train the model with a very small batch, typically from 1 to 5 samples (this is mainly due to the game becoming very slow if, at each step, we used a large batch for training). Each sample in the batch is a transition. Perhaps surprisingly, when training the model, we don’t use the transition that just occured, but sample randomly from the replay buffer. This is done to break the correlation between successive training samples and thereby facilitate learning. The core logic of training a Deep Q-Network is explained above. The nitty gritty details are explained in the Model file in between the code. 

**(7) Check whether game has ended.** If the game has not ended, the algorithm continues to the next iteration. If the game has ended, (i) the game is reset; (ii) the model is trained with a larger batch; (iii) the target net weights are updated with the current policy net weights if needed; (iv) and lastly, the performance is recorded and plotted with the Plotter. 

**Some notoions about the current configuration**

From testing the model repeatedly, I’ve noticed that a relatively simple network works the best. In particular, I use a single layer Feed Forward Netwrok with 256 hidden nodes. Weirdly, I also get the best peformance out of the model by updating the target net weights at the end of each game, which is identical to using policy net in lieu of the target net. One can obviously modify the paremters to his or her liking.

When running the program, please, give the Agent some time to explore the environment. In the current configuration, the performance should pick up shaprly after 60 to 70 played games.


