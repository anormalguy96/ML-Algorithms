# reinforcement_learning.py
# Q-Learning and SARSA from scratch (tabular)

import random


### -----------------
### Q-LEARNING AGENT
### -----------------

class Q_Learning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.states = states
        self.actions = actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {s: {a: 0.0 for a in actions} for s in states}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get)

    def update(self, state, action, reward, next_state):
        max_future = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future - current_q)


### -----------
### SARSA AGENT
### -----------

class SARSA:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {s: {a: 0.0 for a in actions} for s in states}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get)

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_q - current_q)