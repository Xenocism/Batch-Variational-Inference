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

def var(data):
    
    # Means
    m1 = np.mean(data[:,0])
    m2 = np.mean(data[:,1])
    
    # Sums
    s1 = 0.0
    s2 = 0.0

    for i in range(len(data)):
        s1 += ((data[i,0] - m1) ** 2)
        s2 += ((data[i,1] - m2) ** 2)

    # Normalize
    s1 = s1 / len(data)
    s2 = s2 / len(data)

    return np.array([s1, s2])

if __name__ == "__main__":
    
    x = data()
    plt.plot(x[:,0],x[:,1],'o')
    plt.show()
    print(var(x))
