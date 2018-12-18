# Our baseline agent

class BaselineAgent:

    def __init__(self, difference):
        self.difference = difference

    def epsilon_greedy(self, s, eps=.5):
        # We ignore epsilon here as this agent does not get trained

        current_diff = s[-2] # If close price diff is < self.difference: buy
        if current_diff < self.difference:
            return "buy"
        else:
            return "wait"


