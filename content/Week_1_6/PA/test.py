import numpy as np
import matplotlib.pylab as plt
import pandas as pd

A = np.zeros((5, 5))
np.fill_diagonal(A, 5)
A[range(4), range(1, 5)] = 1