from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

def data():

    # Seed a Guassian Mixture Model with three nodes centered around (0,0), (10,10), and (30,30)

    np.random.seed(1)
    model = mixture.GMM(n_components = 3)
    obs1 = np.concatenate((np.random.randn(100, 1),np.random.randn(100, 1)),axis = 1)
    obs2 = np.concatenate((10 + np.random.randn(300, 1), 10 + np.random.randn(300, 1)),axis = 1)
    obs3 = np.concatenate((30 + np.random.randn(200, 1), 30 + np.random.randn(200, 1)), axis = 1)
    obs = np.concatenate((obs1, obs2))
    obs = np.concatenate((obs, obs3))
    model.fit(obs)

    # Sample and display from GMM

    x = model.sample(n_samples = 300)
    return x

if __name__ == "__main__":
    
    x = data()
    plt.plot(x[:,0],x[:,1],'o')
    plt.show()
