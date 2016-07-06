from __future__ import division
from __future__ import print_function

import VI
import math
import matplotlib.pyplot as plt
import numpy as np


class GaussianMixtureModelCAVI(object):

    def __init__(self, x, k, SigSqr):
        
        # Inputs
        self.X = x
        self.K = k
        self.SigSqr = SigSqr

        # Variational Params
        self.phi = np.ones(((len(self.X), self.K, len((self.X).T)))) / self.K
        self.mu = self.Init_Mu(self.X)
        self.var = np.ones((self.K, len((self.X).T)))

    def update(self):
        
        # Phi updates (i: data, j: clusters, k: data dimensionality)
        for i in range(len(self.X)):
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    self.phi[i, j, k] = self.mu[j, k] * (self.X)[i, k] - ((self.mu[j, k] + self.var[j, k]) / 2) # math.exp()
        
        # Normalize Phi
        for i in range(len(self.X)):
            sum = 0.0;
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    sum += self.phi[i, j, k]
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    self.phi[i, j, k] = self.phi[i, j, k] / sum

        # Mu and Var updates
        sum1 = np.zeros((self.K, len((self.X).T)))
        sum2 = np.zeros((self.K, len((self.X).T)))

        for i in range(len(self.X)):
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    sum1[j, k] += self.phi[i, j, k] * self.X[i, k]
                    sum2[j, k] += self.phi[i, j, k] 

        for i in range(self.K):
            for j in range(len((self.X).T)):
                self.mu[i, j] = sum1[i, j] / ((1 / self.SigSqr) + sum2[i, j])
                self.var[i, j] = 1 / ((1 / self.SigSqr) + sum2[i, j])

        # Return ELBO
        

    def Init_Mu(self, x):
        
        # (Temporary) Actual Values
        # Three farthest from each other Hueristic will be here
        return np.array(([0, 0], [10, 10], [30, 30]))
    
    
if __name__ == "__main__":

    data = VI.data()
    k = 3
    s2 = .75 * VI.var(data) # Varience of all the data

    # Create model
    test = GaussianMixtureModelCAVI(x = data, k = k, SigSqr = s2)
    
    # Test and plot
    iter = 100
    data = np.zeros(iter)
    axis = np.linspace(0,iter,1)

    test.update()

    #for i in range(iter):
    #    data[i] = test.update()

    #plt.plot(axis,data,'o')
