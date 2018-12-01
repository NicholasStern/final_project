import numpy as np
from mdp import MDP, TabularQ, Q_learn, greedy

# states: list or set of states
# actions: list or set of actions
# discount_factor: real, greater than 0, less than or equal to 1
# start: specifying initial state

states = # TODO
actions = # TODO
discount_factor = # TODO
start = # TODO

# transition_model: function from (state, action) to return the next state
def transition_model(state, action):
    # TODO

# reward_fn: function from (state, action) to real-valued reward
def reward_fn(state, action):
    # TODO

stock_agent = MDP(states, actions, transition_model, reward_fn, discount_factor)
Q = TabularQ(stock_agent.states, stock_agent.actions)
Q_learn(stock_agent, Q)
print(Q.q)



