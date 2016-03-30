import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

inputData = np.loadtxt('data/train.csv', delimiter=',', skiprows=1)
[m,n] = inputData.shape
plotonly = 500
inputData = inputData[np.random.randint(m,size=plotonly), :]
X = inputData[:, 0: n-1]
Y = inputData[:, n-1]
#try to clear input data
del inputData

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

low_dim_embs = tsne.fit_transform(X)

labels = Y
plt.plot(low_dim_embs)
plt.ylabel(labels)
plt.show()
