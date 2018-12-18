## Approximate Q Learning for Stock Trend Prediction
import numpy as np
import random
from collections import defaultdict

class LinearAQ():

    def __init__(self, mode, actions, states, epsilon, discount, alpha):
        self.mode = mode  # train or test
        self.actions = actions  # list of actions
        self.states = states  # 3D array of states  (# windows, # days, size of state)
        self.epsilon = epsilon  # randomness of actions
        self.discount = discount  # discount factor
        self.alpha = alpha  # learning rate

        # initialize linear model w/ weights dictionary for each action
        self.models = {action: np.zeros(self.states.shape[-1]) for action in self.actions}

    def switch_mode(self, new_mode, new_states, epsilon=0):
        self.mode = new_mode
        self.states = new_states
        self.epsilon = epsilon

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

    def update(self, a, s, sp, r):
        '''
            - Takes in action a, state s, next state sp, and reward r
            - Updates weights in corresponding linear function approximation
        '''
        old_q = self.predict(a, s)
        new_q = max([self.predict(new_action, sp) for new_action in self.actions])

        if a == 'buy':
            self.models['buy'] += self.alpha * (r + self.discount * new_q - old_q) * np.array(s)

        else:
            self.models['wait'] += self.alpha * (r + self.discount * new_q - old_q) * np.array(s)


    def predict(self, a, s):
        '''
            - Takes in action a and state s
            - Performs function approximation to predict q-value
            - Returns q-value prediction
        '''
        if a == 'buy':
            weights = self.models['buy']
            q_val = np.dot(weights, np.array(s))

        else:
            weights = self.models['wait']
            q_val = np.dot(weights, np.array(s))

        return q_val


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
            - Takes in action and returns next state
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
        s = tuple(self.states[self.w][self.t]) # init state
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

        for i, model in enumerate(self.models.values()):
            print('Action {} weights: {}'.format(self.actions[i], model))
        return actions, profit, regret