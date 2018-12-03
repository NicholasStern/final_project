import numpy as np
from mdp import MDP, TabularQ, Q_learn
import pandas as pd

# import history
df = pd.read_csv('histories/amazon.csv')
hist = df.Close - df.Open
hist.apply(np.sign)
hist[hist == -1] = 0  # no price change is considered a decrease
p = 0  # pointer index to history

# states: list or set of states
# actions: list or set of actions
# discount_factor: real, greater than 0, less than or equal to 1
# start: specifying initial state

def gen_states(h):
    # generates possible states for a history window h
    states = []
    for i in range(2**h):
        b = bin(i)[2:]
        l = len(b)
        b = str(0) * (h - l) + b
        states.append(tuple([int(i) for i in b]))

    u_states = [] # uncertain states
    c = 0
    rem = len(states) // 2
    while rem > 0:
        c += 1
        for k in range(rem):
            new_ustate = list(states[k])
            new_ustate[:c] = [-1]*c
            u_states.append(tuple(new_ustate))
        rem = rem // 2

    states.extend(u_states)

    return states

states = gen_states(3)
actions = ['buy', 'wait']
discount_factor = 1
start = (-1, -1, -1)
reward = 2  # reward agent gets for good purchase
wait_penalty = -1  # penalty agent gets for not doing anything

# transition_model: function from (state, action) to return the next state at point "p" in the history
def transition_model(state, action, p):
    if action == 'buy':
        return None  # signifying terminal state has been reached
    else:
        new_state = list(state[1:])
        new_state.append(hist[p])
        return new_state

# reward_fn: function from (state, action) to real-valued reward at point "p" in the history
def reward_fn(state, action, p):
    if action == 'buy' and hist[p]:  # if stock went up after buying
        return reward
    elif action == 'buy' and ~hist[p]:  # if stock went down after buying
        return -reward
    else:
        return wait_penalty


stock_agent = MDP(states, actions, transition_model, reward_fn, discount_factor, p)
Q = TabularQ(stock_agent.states, stock_agent.actions)
Q_learn(stock_agent, Q, eps = 0) # setting eps = 0 means no epsilon-greedy
print(Q.q)



