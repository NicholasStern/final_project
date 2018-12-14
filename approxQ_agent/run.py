import pandas as pd
import numpy as np
from approxQ import ApproxQ

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

def evaluate(data, regret):
    pass

## Initialize Parameters
window_size = 5
train, val, test = gen_data('../histories/apple.csv', window_size)
actions = ['buy', 'wait']
epsilon = 0
discount = 1

## Pass to Approximate Q-Learning Agent
agent = ApproxQ('train', actions, train, epsilon, discount)
train_actions, train_regret = agent.learn()

agent.switch_mode('test', val)
val_actions, val_regret = agent.learn()