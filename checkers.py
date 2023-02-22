import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

scan = int(input("Enter the size you would like the graph: "))
x = np.empty([scan, scan])
i = 0
while i < scan:
    j = 0
    while j < scan:
        if i % 200 < 100:
            if j % 200 < 100:
                x[i][j] = 1
            else:
                x[i][j] = 0
        else:
            if j % 200 < 100:
                x[i][j] = 0
            else:
                x[i][j] = 1
        j += 1
    i += 1

fig, ax = plt.subplots()
i = ax.imshow(x, cmap=cm.gray, interpolation='none')

plt.show()