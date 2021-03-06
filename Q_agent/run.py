import numpy as np
from mdp import MDP, TabularQ, Q_learn, epsilon_greedy
import pandas as pd
from collections import Counter
import argparse
import h5py

def gen_hist(path):
    # imports history from csv
    df = pd.read_csv(path)
    hist = df.Close - df.Open
    hist = hist.apply(np.sign)
    hist[hist == -1] = 0  # no price change is considered a decrease
    hist = [int(h) for h in hist]
    return hist

def gen_states(h):
    # generates possible states for a history window h
    states = [('T')]
    for i in range(2**(h+1)):
        b = bin(i)[2:]
        l = len(b)
        b = str(0) * ((h+1) - l) + b
        states.append(tuple([int(i) for i in b]))

    return states

# states: list or set of states
# actions: list or set of actions
# discount_factor: real, greater than 0, less than or equal to 1
# start: specifying initial state

# read in parameters from command line
parser = argparse.ArgumentParser()

parser.add_argument('-hi', "--history", type=int,
                    help='The history window length')

parser.add_argument('-r', "--reward", type=float, nargs=2,
                    help='The rewards for buy, wait in that order')

parser.add_argument('-e', "--epsilon", type=float,
                    help='Epsilon value for the epsilon greedy algorithm')

parser.add_argument('-d', "--discount", type=float,
                    help='Discount factor')

parser.add_argument('-v', "--verbose", action='store_true',
                    help='Verbosity: if True then will output table of Q values')

args = parser.parse_args()

if args.history:
    hwindow = args.history
else:
    hwindow = 3  # default

if args.reward:
    reward = args.reward[0]  # reward agent gets for good purchase
    wait_penalty = args.reward[1]  # penalty agent gets for not doing anything
else:
    reward = 1  # default
    wait_penalty = -.1  # default

if args.epsilon:
    epsilon = args.epsilon
else:
    epsilon = 0.05  # default

if args.discount:
    discount_factor = args.discount
else:
    discount_factor = 1  # default


#########################################################

hist = gen_hist('../histories/Apple_cleaned.csv')
p = np.copy(hwindow)  # pointer index to history (start at 4th element so we have a history window
states = gen_states(hwindow)
actions = ['buy', 'wait']
start = tuple([hist[i] for i in range(hwindow)])

# transition_model: function from (state, action) to return the next state at point "p+1" in the history
def transition_model(state, action, p):
    if p == len(hist):
        return None
    elif action == 'buy':
        return ('T')  # signifying terminal state has been reached
    else:
        new_state = list(state[1:])
        new_state.append(hist[p])
        return tuple(new_state)

# reward_fn: function from (state, action) to real-valued reward at point "p" in the history
def reward_fn(state, action, p):
    if p == len(hist):  # if we have reached the end of the data
        return None
    elif state == ('T'):  # if terminal state
        return 0
    elif action == 'buy' and hist[p] == 1:  # if stock went up after buying
        return reward
    elif action == 'buy' and hist[p] == 0:  # if stock went down after buying
        return -reward
    else:
        return wait_penalty

def evaluate(hist, actions):
    assert len(actions) == (len(hist) - (hwindow +1))
    actions = list(map(lambda x: 1 if x == "buy" else 0, actions))  # if want to buy if we expect price to increase
    return np.sum(np.array(actions) == np.array(hist[hwindow+1:]))/len(actions)

stock_agent = MDP(states, actions, transition_model, reward_fn, p, hist, hwindow, discount_factor)
Q = TabularQ(stock_agent.states, stock_agent.actions)
Q, actions = Q_learn(stock_agent, Q, iters=2*len(hist[p:-1])+1, eps = epsilon) # setting eps = 0 means no epsilon-greedy

### Text File Output for Readability ###

print('\nParameters of run: hwindow {}, reward {}, epsilon {}, discount {}'.format(hwindow, (reward, wait_penalty),
                                                                                 epsilon, discount_factor))

score = evaluate(hist, actions)
print('Accuracy Score: ', score)

ac = Counter(actions)
purch = ac['buy']/len(actions)
print('Fraction of Purchases: ', purch)

if args.verbose:
    print('Q Table:')
    for key, val in sorted(Q.q.items(),key=lambda x: x[1]):
        print('{}: {}'.format(key, val))

### HDF5 File Output for Computer Readability ###

name = 'hwindow {} reward {} wait_penalty {} epsilon {} discount {}'.format(hwindow, reward, wait_penalty,
                                                                                 epsilon, discount_factor)
data = np.array([score, purch])

with h5py.File("results.hdf5", mode='a') as f:
    try:  # if dataset already exists, overwrite
        curr_data = f[name]
        curr_data[...] = data
    except:  # otherwise create it
        f.create_dataset(name, data=data)
