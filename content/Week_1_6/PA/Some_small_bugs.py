## Some small bugs for you to find!
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = np.linspace(0, 10, 11)

print(a.size())

for i in range(0, a.size):
    a[i] = a[i] +a[i-1]

# Does this look like a sine wave?
plt.plot(a, np.sin(a), label='sin(a)')

c = a+b
