# Baseline
import numpy as np
import pandas as pd
from baseline_agent import BaselineAgent

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


window_size = 10  # days
history_size = 3  # days
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)


def evaluate_agent(agent, states_data):
    """
    This evaluation is based on how much the close price is lower when the
    agent decides to buy compared to the initial price of the time window.
    """
    scores = []
    never_bought_count = 0
    window_size = states_data.shape[1]
    time_bought = np.zeros(window_size + 1)
    for episode in states_data:
        for t, state in enumerate(episode):
            if t == len(episode) - 1:
                # You have to buy at last time frame if didn't buy before
                action = "buy"
                never_bought_count += 1
                time_bought[window_size] += 1
                scores.append(-state[-2])
                break
            else:
                # Get the best action for this state
                action = agent.epsilon_greedy(state, eps=0)
                if action == "buy":
                    scores.append(-state[-2])
                    time_bought[t] += 1
                    break
    score = np.mean(scores)
    proportion_no_action = never_bought_count / len(states_data) * 100
    print("Average score for the agent is {} and doesn't buy in {}% of the cases.".format(score, proportion_no_action))
    for t, c in enumerate(time_bought):
        if t < window_size:
            print("t=%i  Bought %i times." % (t, c))
        else:
            print("Did not buy %i times." % c)
    return score, proportion_no_action, time_bought


evaluate_agent(BaselineAgent(-1), test)
