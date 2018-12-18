import pandas as pd
from collections import Counter
import numpy as np
from NNQ import NNQ

def gen_states(path, window_size, history_size):
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

def evaluate(title, actions, profit, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase Fraction: {:.4f}'.format(title, purch))
    print('{} avg. profit: {:.4f}'.format(title, np.mean(profit)))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 10
history_size = 3
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)
actions = ['buy', 'wait']
epsilon = .3
discount = .9
num_layers = 20
num_units = 100


## Pass to Approximate Q-Learning Agent
agent = NNQ('train', actions, train, epsilon, discount, num_layers, num_units)
train_actions, train_profit, train_regret = agent.learn()
evaluate('train', train_actions, train_profit, train_regret)

agent.switch_mode('val', val)
val_actions, val_profit, val_regret = agent.learn()
evaluate('val', val_actions, val_profit, val_regret)
