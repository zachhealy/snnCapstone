import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = np.empty([1000, 1000])
i = 0
while i < 1000:
    j = 0
    while j < 1000:
        if i % 200 < 100:
            if j % 200 < 100:
                X[i][j] = 1
            else:
                X[i][j] = 0
        else:
            if j % 200 < 100:
                X[i][j] = 0
            else:
                X[i][j] = 1
        j += 1
    i += 1

fig, ax = plt.subplots()
i = ax.imshow(X, cmap=cm.gray, interpolation='none')

plt.show()