import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = 1*np.random.rand(1000,1000)

fig, ax = plt.subplots()
i = ax.imshow(X, cmap=cm.gray, interpolation='none')

plt.show()
