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
    for i in range(2**h):
        b = bin(i)[2:]
        l = len(b)
        b = str(0) * (h - l) + b
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

def gen_episode_states(path, window_size, history_size):
    # read in data
    df = gen_hist(path)

    df_split = np.array([df[i-history_size:i+window_size] for i in range(history_size, len(df) - window_size - 1, window_size)][:-1])

    result_states = []
    for episode in df_split:
        episode_states = []
        for t in range(window_size):
            episode_states.append(tuple(episode[t:t+history_size+1].reshape(-1)))
        result_states.append(episode_states)

    # split into train/val/test (80%, 10%, 10%)
    train = result_states[:int(.8 * len(result_states))]
    val = result_states[int(.8* len(result_states)):int(.9 * len(result_states))]
    test = result_states[int(.9 * len(result_states)):]

    return train, val, test

def gen_states_new_agents(path, window_size, history_size):
    # read in data
    df = pd.read_csv(path)
    df = df[['Open', 'High', 'Low', 'Close']].values

    df_split = np.array([df[i-history_size:i+window_size] for i in range(history_size, len(df) - window_size - 1, window_size)][:-1])
    df_split -= df_split[:,history_size,:][:,np.newaxis,:]

    result_states = []
    for episode in df_split:
        episode_states = []
        for t in range(window_size):
            episode_states.append(np.append(episode[t:t+history_size+1].reshape(-1), t))
        result_states.append(episode_states)

    # split into train/val/test (80%, 10%, 10%)
    train = np.array(result_states[:int(.8 * len(result_states))])
    val = np.array(result_states[int(.8* len(result_states)):int(.9 * len(result_states))])
    test = np.array(result_states[int(.9 * len(result_states)):])

    return train, val, test

window_size = 5
history_size = hwindow - 1

train, val, test = gen_episode_states('../histories/Apple_cleaned.csv', window_size, history_size)
train_new, val_new, test_new = gen_states_new_agents('../histories/Apple_cleaned.csv', window_size, history_size)

def evaluate_agent(agent, states_data, states_new, window_size, verbose=True):
    """
    This evaluation is based on how much the close price is lower when the
    agent decides to buy compared to the initial price of the time window.
    """
    scores = []
    never_bought_count = 0
    time_bought = np.zeros(window_size + 1)
    for episode, episode_new in zip(states_data, states_new):
        for t, (state, state_new) in enumerate(zip(episode, episode_new)):
            if t == len(episode) - 1:
                # You have to buy at last time frame if didn't buy before
                action = "buy"
                never_bought_count += 1
                time_bought[window_size] += 1
                scores.append(-state_new[-2])

                break
            else:
                # Get the best action for this state
                action = epsilon_greedy(agent, state, eps=0)
                if action == "buy":
                    scores.append(-state_new[-2])
                    time_bought[t] += 1
                    break
    score = np.mean(scores)
    proportion_no_action = never_bought_count / len(states_data) * 100

    if verbose:
        print("Average score for the agent is {} and doesn't buy in {}% of the cases.".format(score, proportion_no_action))
        for t, c in enumerate(time_bought):
            if t < window_size:
                print("t=%i  Bought %i times." % (t, c))
            else:
                print("Did not buy %i times." % c)

    return score, proportion_no_action, time_bought

def evaluate_agent_function(agent, states_data, verbose):
    return evaluate_agent(agent, states_data, test_new, window_size, verbose=verbose)

import sys
sys.path.append('..')
from utils import evaluate_agent_advanced

def reset_agent(agent):
    hist = gen_hist('../histories/Apple_cleaned.csv')
    p = np.copy(
        hwindow)  # pointer index to history (start at 4th element so we have a history window
    states = gen_states(hwindow)
    actions = ['buy', 'wait']

    stock_agent = MDP(states, actions, transition_model, reward_fn, p,
                      hist, hwindow, discount_factor)
    agent = TabularQ(stock_agent.states, stock_agent.actions)
    agent, _ = Q_learn(stock_agent, agent, iters=2 * len(hist[p:-1]) + 1,
                       eps=epsilon)  # setting eps = 0 means no epsilon-greedy
    return agent

evaluate_agent_advanced(Q, test, n=100, evaluate_agent_function=evaluate_agent_function, reset_agent_function=reset_agent)
