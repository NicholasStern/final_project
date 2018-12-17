## Approximate Q Learning for Stock Trend Prediction
import numpy as np
import random
from collections import defaultdict
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam



def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, mode, actions, data, epsilon, discount, nfeats, num_layers = 3, num_units = 100):
        self.mode = mode  # train or test
        self.actions = actions  # list of actions
        self.data = data  # market history to walk through in the form of a 2D array
        self.epsilon = epsilon  # randomness of actions
        self.discount = discount  # discount factor
        self.nfeats = nfeats  # number of features
        dim = self.nfeats

        # initialize linear model w/ weights dictionary for each action
        self.models = dict([(a, make_nn(dim, num_layers, num_units)) for a in actions])

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
            regret = choice - best
            # reward = 0
            # if 0 <= regret < 0.005:
            #     reward = 50
            # elif 0.005 <= regret < 0.01:
            #     reward = 40
            # elif 0.01 <= regret < 0.015:
            #     reward = 30
            # elif 0.015 <= regret < 0.02:
            #     reward = 20
            # elif 0.02 <= regret < 0.025:
            #     reward = 10
            # else:
            #     reward = 0
            # return regret, reward
            return regret, regret
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
        self.models[a].fit(X, Y, epochs=1)

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


        return tuple(np.append(self.data[self.w][self.t], self.t))



    def learn(self, iters=100):
        '''
            - Takes in state and epsilon and returns action
        '''
        self.w = 0  # window index
        self.t = 0  # timestep index within window
        s = tuple(np.append(self.data[self.w][self.t], self.t)) # init state
        actions = []
        regrets = []
        for _ in range(iters):
            a = self.epsilon_greedy(s, self.epsilon)
            actions.append(a)

            regret, reward = self.reward(a)
            if a == 'buy':
                regrets.append(regret)

            s_prime = self.transition(a)

            if s_prime is None:
                break

            if self.mode == 'train':
                self.update(a, s, s_prime, reward)

            s = s_prime

        return actions, regrets

import pandas as pd
from collections import Counter

def gen_data(path, window_size):
    # read in history
    df = pd.read_csv(path)
    df = df[['Open', 'High', 'Low', 'Close']].values

    # segment into time windows
    df_split = np.array_split(df, df.shape[0]//window_size)

    # fix the size of the first window if split is uneven
    df_split[0] = df_split[0][len(df_split[0])-window_size:]

    # split into train/val/test (80%, 10%, 10%)
    train = df_split[:int(.8 * len(df_split))]
    val = df_split[int(.8* len(df_split)):int(.9 * len(df_split))]
    test = df_split[int(.9 * len(df_split)):]

    return train, val, test

def evaluate(title, actions, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase Fraction: {:.4f}'.format(title, purch))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 5
train, val, test = gen_data('../histories/apple.csv', window_size)
actions = ['buy', 'wait']
epsilon = .3
discount = .9
nfeats = 5
num_layers = 20
num_units = 100


## Pass to Approximate Q-Learning Agent
agent = NNQ('train', actions, train, epsilon, discount, nfeats, num_layers, num_units)
train_actions, train_regret = agent.learn()
evaluate('train', train_actions, train_regret)

agent.switch_mode('val', val)
val_actions, val_regret = agent.learn()
evaluate('val', val_actions, val_regret)




# def make_nn(state_dim, num_hidden_layers, num_units):
#     model = Sequential()
#     model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
#     for i in range(num_hidden_layers-1):
#         model.add(Dense(num_units, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mse', optimizer=Adam())
#     return model
#
# class NNQ:
#     def __init__(self, actions, num_layers, num_units, epochs=1):
#         self.actions = actions
#         self.epochs = epochs
#         dim = self.state2vec(self.states[0]).shape[1]
#         self.models = dict([(a, make_nn(dim, num_layers, num_units)) for a in actions])  # Your code here
#
#     def get(self, s, a):
#         # Your code here
#         return self.models[a].predict(self.state2vec(s))
#
#     def update(self, data, lr, epochs=1):
#         # Your code here
#         for a in self.actions:
#             data_a = []
#             for ata in data:
#                 if ata[1] == a:
#                     data_a.append(ata)
#             if data_a != []:
#                 X = np.vstack([self.state2vec(s[0]) for s in data_a])
#                 Y = np.vstack([t[-1] for t in data_a])
#                 self.models[a].fit(X, Y, epochs=epochs)
#
# def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5):
#     # Your code here
#     # raise NotImplementedError('Q_learn')
#
#     s = mdp.init_state()
#     for i in range(iters):
#         # include this line in the iteration, where i is the iteration number
#         if mdp.terminal(s):
#             a = epsilon_greedy(q, s, eps)
#             r, s_prime = mdp.sim_transition(s, a)
#             t = r
#             q.update([(s, a, t)], lr)
#             s = s_prime
#         else:
#             a = epsilon_greedy(q, s, eps)
#             r, s_prime = mdp.sim_transition(s, a)
#             t = r + mdp.discount_factor * value(q, s_prime)
#             q.update([(s, a, t)], lr)
#             s = s_prime
#     return q
#
#
# class MDP:
#
#     def __init__(self, mode, actions, data, epsilon, discount):
#         self.mode = mode  # train or test
#         self.actions = actions  # list of actions
#         self.data = data  # market history to walk through in the form of a 2D array
#         self.epsilon = epsilon  # randomness of actions
#         self.discount = discount  # discount factor
#         #
#         # # initialize linear model w/ weights dictionary for each action
#         # self.models = {action: defaultdict(int) for action in self.actions}
#
#     def switch_mode(self, new_mode, new_data):
#         self.mode = new_mode
#         self.data = new_data
#
#     # Given a state, return True if the state should be considered to
#     # be terminal.
#     def terminal(self, state):
#         #TODO
#
#     # Choose a state to start over
#     def init_state(self):
#         #TODO
#
#     def sim_transition(self, s, a):
#         if self.terminal(s):
#             return self.reward_fn(s, a, self.p), self.init_state()
#         else:
#             self.p += 1
#             return self.reward_fn(s, a, self.p), self.transition_model(s, a, self.p)
#
#     def reward(self, a):
#         '''
#             - Calculates reward to give agent based on action a at timestep t in window w
#             - Reward is based on regret
#         '''
#
#         if a == 'buy' or self.t == (len(self.data[self.w])-1):  # force agent to buy at end of time frame
#             r = (self.data[self.w][self.t] - min(self.data[self.w]))/min(self.data[self.w])
#             return r
#         else:
#             return 0
#
#
#     def transition(self, a):
#         '''
#             - Takes in t, w and returns next state
#         '''
#
#         self.t += 1
#         if self.t == len(self.data[self.w]) or a == 'buy':
#             self.t = 0
#             self.w += 1
#             if self.w == len(self.data):
#                 return None  # We are finished traversing data
#
#         return tuple(self.data[self.w][self.t])
#
#
#
#     def learn(self, iters=100):
#         '''
#             - Takes in state and epsilon and returns action
#         '''
#         self.w = 0  # window index
#         self.t = 0  # timestep index within window
#         s = tuple(self.data[self.w][self.t]) # init state
#         actions = []
#         regret = []
#         for _ in range(iters):
#             a = self.epsilon_greedy(s, self.epsilon)
#             actions.append(a)
#
#             r = self.reward(a)
#             regret.append(r)
#
#             s_prime = self.transition(a)
#
#             if s_prime is None:
#                 break
#
#             if self.mode == 'train':
#                 self.update(a, s, s_prime, r)
#
#             s = s_prime
#
#         return actions, regret
#
# # The q function is typically an instance of NNQ, implemented as a
# # dictionary mapping (s, a) pairs into Q values
#
#
# # Given a state, return the value of that state, with respect to the
# # current definition of the q function
# def value(q, s):
#     return max(q.get(s, a) for a in q.actions)
#
# def argmax(l, f):
#     """
#     @param l: C{List} of items
#     @param f: C{Procedure} that maps an item into a numeric score
#     @returns: the element of C{l} that has the highest score
#     """
#     vals = [f(x) for x in l]
#     return l[vals.index(max(vals))]
#
# # Given a state, return the action that is greedy with respect to the
# # current definition of the q function
# def greedy(q, s):
#     return argmax(q.actions, lambda a: q.get(s, a))
#
# # return a randomly selected element from the list
# def randomly_select(alist):
#     return alist[random.randint(0,len(alist)-1)]
#
# def epsilon_greedy(q, s, eps = 0.5):
#     if random.random() < eps:  # True with prob eps, random action
#         return randomly_select(q.actions)
#     else:                   # False with prob 1-eps, greedy action
#         return greedy(q, s)
#
#
#
#
# stock_agent = MDP(states, actions, transition_model, reward_fn, p, hist, hwindow, discount_factor)
# Q = NNQ(stock_agent.states, stock_agent.actions, num_layers, num_units,
#                 epochs=1)
# q = Q_learn(stock_agent, Q, iters=2*len(hist[p:-1])+1, eps = epsilon)