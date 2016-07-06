from __future__ import division
from __future__ import print_function

import VI
import math
from scipy.special import digamma
import matplotlib.pyplot as plt
import numpy as np


class GaussianMixtureModelCAVI(object):

    def __init__(self, x, k, SigSqr):
        
        # Inputs
        self.X = x
        self.K = k
        self.SigSqr = SigSqr

        """ ~ X is the given data set, a 2D array of doubles
        ~ K is the expected number of clusters to assign Guassian models to in the VI algorithm
        ~ SigSqr is the Variation of the given data set (an array with one value per dimension) """

        # Variational Params
        self.phi = np.ones(((len(self.X), self.K, len((self.X).T)))) / self.K
        self.mu = self.Init_Mu(self.X)
        self.var = np.ones((self.K, len((self.X).T)))

        """ ~ Phi is a 3D array of cluster assignment probabilities, with an [0,1] float for each data point, 
        in each dimension per expected cluster
        ~ Mu is the mean of each expected Guassian in the misture model  
        ~ Var is the variation of each expected Guassian in the mixture model """

    def update(self):
        
        # Phi updates (i: data, j: clusters, k: data dimensionality)
        for i in range(len(self.X)):
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    self.phi[i, j, k] = math.exp(self.mu[j, k] * (self.X)[i, k] - (((self.mu[j, k] ** 2) + (self.var[j, k]) ** 2)) / 2)
        
        # Normalize Phi (each dim should sum to 1)
        for i in range(len(self.X)):
            sum = 0.0;
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    sum += self.phi[i, j, k]
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    self.phi[i, j, k] = self.phi[i, j, k] / (.5 * sum)

        # Mu and Var updates
        sum1 = np.zeros((self.K, len((self.X).T)))
        sum2 = np.zeros((self.K, len((self.X).T)))

        # initialize summations
        for i in range(len(self.X)):
            for j in range(self.K):
                for k in range(len((self.X).T)):
                    sum1[j, k] += self.phi[i, j, k] * self.X[i, k]
                    sum2[j, k] += self.phi[i, j, k] 

        for i in range(self.K):
            for j in range(len((self.X).T)):
                (self.mu)[i, j] = (sum1[i, j] / ((1 / self.SigSqr[j]) + sum2[i, j]))
                (self.var)[i, j] = (1 / ((1 / self.SigSqr[j]) + sum2[i, j]))

        # Return ELBO
        # return (math.log() + )

    def Init_Mu(self, x):
        
        # (Temporary) Actual Values
        # Three farthest from each other Hueristic will be here
        return np.array(([0.0, 0.0], [10.0, 10.0], [30.0, 30.0]))
    
    
if __name__ == "__main__":

    data = VI.data()        # Samples from the custom GMM in VI.py
    k = 3                   # Three expected clusters in this custom data set
    s2 = .75 * VI.var(data) # Varience of all the data

    # Create model
    test = GaussianMixtureModelCAVI(x = data, k = k, SigSqr = s2)
    
    # Test and plot the Coordinate Ascent
    iter = 100
    data = np.zeros(iter)
    axis = np.linspace(0,iter,1)

    test.update()

    #for i in range(iter):
    #    data[i] = test.update()

    #plt.plot(axis,data,'o')
