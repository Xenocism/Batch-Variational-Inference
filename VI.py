from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

# Seed a Guassian Mixture Model with three nodes centered around 0, 10, and 30

np.random.seed(1)
model = mixture.GMM(n_components = 3)
obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))
obs = np.concatenate((obs, 30 + np.random.randn(200, 1)))
model.fit(obs)

# Sample and display from GMM

x = model.sample(n_samples = 400)
plt.plot(x, 'o')
#plt.show()

