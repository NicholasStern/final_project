import random
from copy import deepcopy

class MDP:

    def __init__(self, states, actions, transition_model, reward_fn, p, hist, hwindow,
                     discount_factor = 1.0):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.p = deepcopy(p)
        self.hist = hist
        self.hwindow = hwindow

    # Given a state, return True if the state should be considered to
    # be terminal.
    def terminal(self, state):
        if state == ('T'):
            return True
        else:
            return False

    # Choose a state to start over
    def init_state(self):
        new_init_state = tuple([self.hist[self.p-i] for i in range(0, self.hwindow+1)])
        return new_init_state

    # Simulate a transition from state s, given action a at point "p" in history.
    # Return reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        if self.terminal(s):
            return self.reward_fn(s, a, self.p), self.init_state()
        else:
            self.p += 1
            return self.reward_fn(s, a, self.p), self.transition_model(s, a, self.p)


# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values


# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    return max(q.get(s, a) for a in q.actions)

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

# Given a state, return the action that is greedy with respect to the
# current definition of the q function
def greedy(q, s):
    return argmax(q.actions, lambda a: q.get(s, a))

# return a randomly selected element from the list
def randomly_select(alist):
    return alist[random.randint(0,len(alist)-1)]

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        return randomly_select(q.actions)
    else:                   # False with prob 1-eps, greedy action
        return greedy(q, s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        for s,a,t in data:
            self.q[(s,a)] = (1-lr)*self.q[(s,a)] + lr*t


def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5):
    actions = []
    s = mdp.init_state()
    for _ in range(iters):
        if mdp.terminal(s):
            a = epsilon_greedy(q, s, eps)
            r, s_prime = mdp.sim_transition(s, a)
            t = 0
            q.update([(s, a, t)], lr)
            s = s_prime
        else:
            a = epsilon_greedy(q, s, eps)
            actions.append(a)
            r, s_prime = mdp.sim_transition(s, a)
            if s_prime is None:
                break
            t = r + mdp.discount_factor * value(q, s_prime)
            q.update([(s, a, t)], lr)
            s = s_prime
    return q, actions[:-1]  # knock off last one bc there's no way to validate it
