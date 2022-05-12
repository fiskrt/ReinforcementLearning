import numpy as np

class Agent(object):
    """The world's best agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05
        self.Q = np.zeros((self.state_space, self.action_space))
       # self.Q = np.random.rand(self.state_space, self.action_space)
        #print(self.Q)
        self.gamma = 0.95
       # self.learning_rate = 20 
        self.learning_rate = 0.8
        self.state = None

    def observe(self, observation, reward, done):
#        self.learning_rate += 0.0001
        update = reward + ((1-done)*self.gamma * np.max(self.Q[observation,:]))\
                - self.Q[self.state, self.action]
        self.Q[self.state, self.action] += update * self.learning_rate 
#        self.learning_rate = self.learning_rate - 0.00001 
#        if self.learning_rate < 0.001:
#            self.learning_rate = 0.001 
        self.state = observation
        if done:
            self.state = None


    def act(self, observation):
        if self.state is None:
            self.state = observation
        if np.random.uniform(0, 1) < self.epsilon:
            self.action = np.random.randint(self.action_space)
        else:
            self.action = np.random.choice(np.flatnonzero(self.Q[self.state] == self.Q[self.state].max()))
        return self.action


    def optimal(s):
        'return the optimal policy'
        self.action = 0
        F = np.asarray([[0.20450739, 0.1757205,  0.14932901, 0.15008981],
            [0.07848338, 0.08122618, 0.07017175, 0.15007216],
            [0.16028551, 0.11294197, 0.10027242, 0.98676489],
            [0.03443955, 0.00945665, 0.01937776, 0.03487486],
            [0.24503197, 0.17014154 ,0.16518426 ,0.11128652],
            [0,         0,         0,         0],
            [0.08092551, 0.09443288, 0.28675238, 0.03931354],
            [0,         0,         0,         0        ],
            [0.16361758, 0.14135972, 0.18516105, 0.3250894 ],
            [0.25236009, 0.44606372, 0.23427509, 0.24711224],
            [0.4820822,  0.27619719, 0.27288034, 0.17525885],
            [0,         0,         0,         0        ],
            [0,         0,         0,         0        ],
            [0.26967298, 0.34719115, 0.52812726, 0.28566532],
            [0.56172192, 0.77441571, 0.6188472,  0.62407425],
            [0,         0 ,        0,         0        ]])

        if np.random.uniform(0, 1) < self.epsilon:
            self.action = np.random.randint(self.action_space)
        else:
            return np.argmax(F[observation])
 
