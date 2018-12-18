# Baseline
import sys
sys.path.append('..')
from utils import gen_states, evaluate_agent_advanced
from baseline_agent import BaselineAgent

window_size = 10  # days
history_size = 3  # days
train, val, test = gen_states('../histories/Apple_cleaned.csv', window_size, history_size)

evaluate_agent_advanced(BaselineAgent(-1), test)
