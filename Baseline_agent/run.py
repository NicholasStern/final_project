# Baseline
import sys
sys.path.append('..')
from utils import gen_states, evaluate_agent
from baseline_agent import BaselineAgent

window_size = 10  # days
history_size = 3  # days
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)

print('\nTraining Evaluation:')
evaluate_agent(BaselineAgent(-1), train)

print('\nValidation Evaluation:')
evaluate_agent(BaselineAgent(-1), val)
