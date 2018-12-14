## Approximate Q Learning for Stock Trend Prediction
import numpy as np
import random
from collections import defaultdict

class ApproxQ():

    def __init__(self, mode, actions, data, epsilon, discount):
        self.mode = mode  # train or test
        self.actions = actions  # list of actions
        self.data = data  # market history to walk through in the form of a 2D array
        self.epsilon = epsilon  # randomness of actions
        self.discount = discount  # discount factor

        # initialize linear model w/ weights dictionary for each action
        self.models = {action: defaultdict(int) for action in self.actions}

    def switch_mode(self, new_mode, new_data):
        self.mode = new_mode
        self.data = new_data

    def reward(self, a):
        '''
            - Calculates reward to give agent based on action a at timestep t in window w
            - Reward is based on regret
        '''

        if a == 'buy' or self.t == (len(self.data[self.w])-1):  # force agent to buy at end of time frame
            choice = self.data[self.w][self.t][-1]  # close price on chosen day
            best = min([x[-1] for x in self.data[self.w]])  # best close price
            r = (choice - best)/best
            return r
        else:
            return 0

    def update(self, a, s, sp, r):
        '''
            - Takes in action a, state s, next state sp, and reward r
            - Updates weights in corresponding linear function approximation
        '''
        pass

    def predict(self, a, s):
        '''
            - Takes in action a and state s
            - Performs function approximation to predict q-value
            - Returns q-value prediction
        '''
        pass


    def epsilon_greedy(self, s, eps=0.5):
        '''
            - Takes in state and epsilon and returns action
        '''
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

        self.t += 1
        if self.t == len(self.data[self.w]) or a == 'buy':
            self.t = 0
            self.w += 1
            if self.w == len(self.data):
                return None  # We are finished traversing data

        return tuple(self.data[self.w][self.t])



    def learn(self, iters=100):
        '''
            - Takes in state and epsilon and returns action
        '''
        self.w = 0  # window index
        self.t = 0  # timestep index within window
        s = tuple(self.data[self.w][self.t]) # init state
        actions = []
        regret = []
        for _ in range(iters):
            a = self.epsilon_greedy(s, self.epsilon)
            actions.append(a)

            r = self.reward(a)
            regret.append(r)

            s_prime = self.transition(a)

            if s_prime is None:
                break

            if self.mode == 'train':
                self.update(a, s, s_prime, r)

            s = s_prime

        return actions, regret