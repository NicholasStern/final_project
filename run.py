import numpy as np
from mdp import MDP, TabularQ, Q_learn
import pandas as pd
from collections import Counter

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

hist = gen_hist('histories/amazon.csv')
hwindow = 3
p = np.copy(hwindow)  # pointer index to history (start at 4th element so we have a history window
states = gen_states(hwindow)
actions = ['buy', 'wait']
discount_factor = 1
start = tuple([hist[i] for i in range(hwindow)])
reward = 1  # reward agent gets for good purchase
wait_penalty = -.1  # penalty agent gets for not doing anything

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
    if len(state) == 1:  # if terminal state
        return 0
    elif p == len(hist):  # if we have reached the end of the data
        return None
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
Q, actions = Q_learn(stock_agent, Q, iters=2*len(hist[p:-1]), eps = 0) # setting eps = 0 means no epsilon-greedy
for key, val in sorted(Q.q.items(),key=lambda x: x[1]):
    print('{}: {}'.format(key, val))

print('Accuracy Score: ', evaluate(hist, actions))
ac = Counter(actions)
print('Fraction of Purchases: ', ac['buy']/len(actions))