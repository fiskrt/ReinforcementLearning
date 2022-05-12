import numpy as np

class Agent(object):
    """The world's best agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05
      #  self.Q = np.ones((self.state_space, self.action_space))
        self.Q = np.random.rand(self.state_space, self.action_space)
      #  self.Q[-1,:] = 0
        self.gamma = 0.95
        self.learning_rate = 0.4
        
        self.s = None 
        self.a = None

    def observe(self, s_prime, reward, done):
        a_prime = self.eps_greedy(s_prime)
         
        update = reward+(1-done)*self.gamma*self.Q[s_prime, a_prime]-self.Q[self.s, self.a]
        self.Q[self.s, self.a] += self.learning_rate * update 
        self.s = s_prime
        self.a = a_prime
        if done:
            self.s = None
            self.a = None
    
    def eps_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_space)
        return np.random.choice(np.flatnonzero(self.Q[s,:] == self.Q[s,:].max()))

    def act(self, s_prime):
        if self.s is None:
            # Initialize, ran once per reset.
            self.s = s_prime
            self.a = self.eps_greedy(self.s)
        return self.a

