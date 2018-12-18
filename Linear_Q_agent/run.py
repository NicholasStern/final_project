import pandas as pd
import numpy as np
from approxQ import ApproxQ
from collections import Counter

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



    # important check
    # data = [train, val, test]
    # for dataset in data:
    #     for w in dataset:
    #         assert len(w) == window_size

    return train, val, test


def evaluate(title, actions, regret):
    ac = Counter(actions)
    purch = ac['buy'] / len(actions)

    print('{} purchase Fraction: {:.4f}'.format(title, purch))
    print('{} avg. regret: {:.4f}'.format(title, np.mean(regret)))

## Initialize Parameters
window_size = 10  # days
history_size = 3  # days
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)
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