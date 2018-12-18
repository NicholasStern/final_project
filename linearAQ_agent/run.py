import sys
sys.path.append('..')
from utils import gen_states, evaluate_agent_advanced
import pandas as pd
import numpy as np
from linearAQ import LinearAQ
from collections import Counter

def evaluate(title, actions, profit, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase Fraction: {:.4f}'.format(title, purch))
    print('{} avg. profit: {:.4f}'.format(title, np.mean(profit)))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 5  # days
history_size = 3  # days
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)
actions = ['buy', 'wait']
epsilon = .05
discount = 1
alpha = .1

## Pass to Approximate Q-Learning Agent
agent = LinearAQ('train', actions, train, epsilon, discount, alpha)
train_actions, train_profit, train_regret = agent.learn()
evaluate('train', train_actions, train_profit, train_regret)

agent.switch_mode('test', val)
val_actions, val_profit, val_regret = agent.learn()
evaluate('val', val_actions, val_profit, val_regret)

agent.switch_mode('train', train)
evaluate_agent_advanced(agent, test)
