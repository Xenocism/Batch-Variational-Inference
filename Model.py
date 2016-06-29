from __future__ import division
from __future__ import print_function

from VI import data
import matplotlib.pyplot as plt
import numpy as np

class GaussianMixtureModelCAVI(object):

    def __init__(self, x, K, SigSqr):
        
        # Inputs
        self.X = x
        self.K = K
        self.SigSqr = SigmaSqr

        # Variational Params
        self.phi = (np.ones((len(self.X)), self.K) / self.K)
        self.mu = Init_Mu(self.X)
        self.var = np.ones(self.K)

    def update(self):
        
        # Phi updates
        for i in range(len(self.X)):
            for j in range(len((self.X).T))):
                self.phi[i, j] = self.mu[j] - 

        # Mu and Var updates
        

    def Init_Mu(self, x):
        
        # Actual Values
        return np.array(([0, 0], [10, 10], [30, 30]))
    
    
if __name__ == "__main__":

    data = VI.data
    k = 3
    s2 = .75 * #varience of all the data

    # create model
    test = GaussianMixtureModelCAVI(x = data, K = k, SigSqr = s2)
    
    # test and plot
    iter = 100
    data = np.zeros(iter)
    axis = np.linspace(0,iter,1)

    for i in range(iter):
        data[i] = test.update()

    plt.plot(axis,data,'o')
