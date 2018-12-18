import pandas as pd
import numpy as np
from approxQ import ApproxQ
from collections import Counter

def gen_states(path, window_size, history_size):
    # read in data
    df = pd.read_csv(path)
    df = df[['Open', 'High', 'Low', 'Close']].values

    # segment into time windows
    df_split = np.array_split(df[:df.shape[0] // window_size * window_size], df.shape[0] // window_size)

    # split into train/val/test (80%, 10%, 10%)
    train = df_split[:int(.8 * len(df_split))]
    val = df_split[int(.8* len(df_split)):int(.9 * len(df_split))]
    test = df_split[int(.9 * len(df_split)):]

    # important check
    data = [train, val, test]
    for dataset in data:
        for w in dataset:
            assert len(w) == window_size

    # generate states
    for i in range(history_size-1, len(train)*window_size):



    return train, val, test


def evaluate(title, actions, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase Fraction: {:.4f}'.format(title, purch))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 20  # days
history_size = 3  # days
train, val, test = gen_data('../histories/Apple_cleaned.csv', window_size)
actions = ['buy', 'wait']
epsilon = 0
discount = 1
alpha = .1
nfeats = 5

## Pass to Approximate Q-Learning Agent
agent = ApproxQ('train', actions, train, epsilon, discount, alpha, nfeats, history_size)
train_actions, train_regret = agent.learn()
evaluate('train', train_actions, train_regret)

agent.switch_mode('test', val)
val_actions, val_regret = agent.learn()
evaluate('val', val_actions, val_regret)