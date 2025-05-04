# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr
import rospy

class OUNoise:
    def __init__(self,mu , sigma = 0.15 , theta = 0.2, dt = 1e-2 , x0 = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * nr.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def decay_sigma(self, decay_factor=0.995, min_sigma=0.01):
        # Decay sigma (element-wise if sigma is an array)
        self.sigma = np.clip(self.sigma * decay_factor, min_sigma, None)

class OUActionNoise:
    def __init__(self, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = np.zeros(2)
        self.std_dev = np.array([0.2 , 0.2])
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def noise(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

"""class OUActionNoise:
    def __init__(self, theta=0.15, sigma=0.2, dt=0.01):
        self.mu = np.array([0.2 , 0.2])
        self.theta = theta
        self.sigma = sigma  # Can be scalar or array (per-dimension)
        self.dt = dt
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state

    def decay_sigma(self, decay_factor=0.995, min_sigma=0.01):
        # Decay sigma (element-wise if sigma is an array)
        if isinstance(self.sigma, (list, np.ndarray)):
            self.sigma = np.clip(self.sigma * decay_factor, min_sigma, None)
        else:
            self.sigma = max(self.sigma * decay_factor, min_sigma)"""
    
class GaussianNoise:
    def __init__(self,action_dimension):
        self.action_dimension = action_dimension
        self.state = np.ones(self.action_dimension)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) 

    def noise(self):
        #self.state[0] = np.random.normal(loc=0.0, scale=0.15)
        #self.state[1] = np.random.normal(loc=0.0, scale=0.7)
        self.state[0] = np.random.normal(loc=0.0, scale=0.2)
        self.state[1] = np.random.normal(loc=0.0, scale=0.2)
        return self.state

"""class GaussianNoise:
    def __init__(self, action_dimension):
        self.action_dimension = action_dimension
        self.max_lin_noise = 0.5
        self.max_angular_noise = 1.9
        self.min_linear_noise = 0.05
        self.min_angular_noise = 0.1
        self.noise_decay = 0.99
        self.current_noise_scale = 1.0
        
    def reset(self):
        self.current_noise_scale = 1.0
    
    def noise(self):
        # Generate Gaussian noise with decaying amplitude
        noise = np.random.normal(
            loc=0, 
            scale=self.current_noise_scale * self.max_lin_noise, 
            size=self.action_dimension
        )
        
        # Clip noise to ensure it stays within specified bounds
        noise = np.clip(
            noise, 
            [-self.max_lin_noise, -self.max_angular_noise],
            [self.max_lin_noise, self.max_angular_noise]
        )
        
        # Decay the noise scale
        self.current_noise_scale *= self.noise_decay
        
        # Ensure noise doesn't go below minimum threshold
        noise = np.maximum(
            noise, 
            [self.min_linear_noise, self.min_angular_noise]
        )
        
        return noise"""
    

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()