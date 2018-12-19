import sys
sys.path.append('..')
from utils import gen_states, evaluate_agent_advanced
from collections import Counter
import numpy as np
from NNQ import NNQ


def evaluate(title, actions, profit, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase fraction: {:.4f}'.format(title, purch))
    print('{} avg. profit: {:.4f}'.format(title, np.mean(profit)))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 5
history_size = 3
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)
actions = ['buy', 'wait']
epsilon = .05
discount = 1
num_layers = 2
num_units = 100


## Pass to Approximate Q-Learning Agent
agent = NNQ('train', actions, train, epsilon, discount, num_layers, num_units)
train_actions, train_profit, train_regret = agent.learn()
evaluate('train', train_actions, train_profit, train_regret)

agent.switch_mode('val', val)
val_actions, val_profit, val_regret = agent.learn()
evaluate('val', val_actions, val_profit, val_regret)

