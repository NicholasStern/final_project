## Approximate Q Learning for Stock Trend Prediction
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, mode, actions, states, epsilon, discount, num_layers = 3, num_units = 100):
        self.mode = mode  # train or test
        self.actions = actions  # list of actions
        self.states = states  # market history to walk through in the form of a 2D array
        self.epsilon = epsilon  # randomness of actions
        self.discount = discount  # discount factor
        self.nfeats = self.states.shape[-1]
        self.num_layers = num_layers
        self.num_units = num_units

        # initialize linear model w/ weights dictionary for each action
        self.models = dict([(a, make_nn(self.nfeats, self.num_layers, self.num_units)) for a in self.actions])

    def switch_mode(self, new_mode, new_states):
        self.mode = new_mode
        self.states = new_states

    def reset(self):
        '''
            - resets the weights of the agent (for repeated training and testing situations)
        '''
        self.models = dict([(a, make_nn(self.nfeats, self.num_layers, self.num_units)) for a in self.actions])

    def reward(self, a):
        '''
            - Calculates reward to give agent based on action a at timestep t in window w
            - Reward is based on regret
        '''

        if a == 'buy' or self.t == (len(self.states[self.w])-1):  # force agent to buy at end of time frame
            choice = self.states[self.w][self.t][-2] # difference in price from chosen day to initial day
            best = min([x[-2] for x in self.states[self.w]])  # best close price
            regret = choice - best
            return -choice, regret
        else:
            return 0, 0

    def value(self, s):
        return max(self.predict(a, s) for a in self.actions)

    def update(self, a, s, sp, r):
        '''
            - Takes in action a, state s, next state sp, and reward r
            - Training on your network for action a, on a data set made up of a single data point, where the input is
            - s and the desired output is t
        '''
        t = r + self.discount * self.value(sp)
        X = np.array(s).reshape(1,self.nfeats)
        Y = np.array(t)
        self.models[a].fit(X, Y, epochs=1, verbose=0)

    def predict(self, a, s):
        '''
            - Takes in action a and state s
            - Performs function approximation to predict q-value
            - Returns q-value prediction
        '''

        return self.models[a].predict(np.array(s).reshape(1, self.nfeats))


    def epsilon_greedy(self, s, eps=0.5):
        '''
            - Takes in state and epsilon and returns action
        '''
        if self.mode == 'test':
            eps = 0
            
        if random.random() < eps:  # True with prob eps, random action
            return self.actions[random.randint(0,len(self.actions)-1)]
        else:  # False with prob 1-eps, greedy action
            q_vals = np.zeros(len(self.actions))
            for i, a in enumerate(self.actions):

                q_vals[i] = self.predict(a, s)

            return self.actions[q_vals.argmax()]

    def transition(self, a):
        '''
            - Takes in t, w and returns next state
        '''

        self.t += 1  # move to next day
        if self.t == len(self.states[self.w]) or a == 'buy':
            self.t = 0
            self.w += 1
            if self.w == len(self.states):
                return None  # We are finished traversing state space

        return tuple(self.states[self.w][self.t])  # return next state



    def learn(self, iters=100):
        '''
            - Takes in state and epsilon and returns action
        '''
        self.w = 0  # window index
        self.t = 0  # timestep index within window
        s = tuple(self.states[self.w][self.t])  # init state
        actions = []
        profit = []
        regret = []
        for _ in range(iters):
            a = self.epsilon_greedy(s, self.epsilon)
            actions.append(a)

            r, reg = self.reward(a)
            if a == 'buy':
                profit.append(r)
                regret.append(reg)

            s_prime = self.transition(a)

            if s_prime is None:
                break

            if self.mode == 'train':
                self.update(a, s, s_prime, r)

            s = s_prime

        return actions, profit, regret