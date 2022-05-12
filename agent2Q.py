import numpy as np

class Agent(object):
    """The world's 2nd best agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05 
        self.Q1 = np.random.rand(self.state_space, self.action_space)
        self.Q2 = np.random.rand(self.state_space, self.action_space)
#        self.Q1 = np.zeros((self.state_space, self.action_space))
#        self.Q2 = np.zeros((self.state_space, self.action_space))
        self.gamma = .95
#        self.learning_rate = 0.1
        self.learning_rate = 10
        self.state = None

    @property 
    def Q(self):
        return 0.5*(self.Q1+self.Q2)

    def observe(self, observation, reward, done):
        self.learning_rate += 0.0001
        if np.random.uniform(0, 1) < 0.5:
            update = reward + ((1-done)*self.gamma * 
                 self.Q2[observation,np.argmax(self.Q1[observation,:])]
                 - self.Q1[self.state, self.action]
            )
            self.Q1[self.state, self.action] += update / self.learning_rate
        else:
            update = reward + ((1-done)*self.gamma * 
                 self.Q1[observation, np.argmax(self.Q2[observation,:])]
                 - self.Q2[self.state, self.action]
            )
            self.Q2[self.state, self.action] += update / self.learning_rate 
#        self.state = observation
#        if done:
#            self.state = None

    def act(self, observation):
        self.state = observation
        if np.random.uniform(0, 1) < self.epsilon:
            self.action = np.random.randint(self.action_space)
        else:
            self.action = np.random.choice(np.flatnonzero(self.Q[self.state,:] == self.Q[self.state,:].max()))
        return self.action
