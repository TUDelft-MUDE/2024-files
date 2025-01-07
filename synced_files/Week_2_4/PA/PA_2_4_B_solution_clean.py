import numpy as np
import matplotlib.pyplot as plt
A = np.array([1, 20, 300, 1, 2, 3])
print(A)
B = np.reshape(A, (2, 3))
print(B)
B.mean(axis=0)
B.mean(axis=1)
B.std(axis=1)
coins = [46, 28, 16, 27,
         22, 24, 31, 12,
         32, 36, 12, 0,
         41, 27, 21, 26,
         21, 19, 18, 35,
         14, 34, 8, 0,
         53, 34, 23, 35,
         28, 26, 18, 13,
         12, 14, 34, 0]
coins_matrix = np.reshape(coins, (3, 12))
print(coins_matrix)
np.set_printoptions(precision=1)
print(f'The average number of coins spent per month is: {np.mean(coins):.1f}')
print('The average number of coins spent per month for each year is:', coins_matrix.mean(axis=1))
print(f'The average number of coins spent each september: {coins_matrix.mean(axis=0)[0]:.1f}')
print(f'The average number of coins spent each january: {coins_matrix.mean(axis=0)[4]:.1f}')
print(f'Max coins spent in any month: {max(coins_matrix.max(axis=0)):.1f}')
print(f'Max coins spent in any year: {max(coins_matrix.sum(axis=1)):.1f}')
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(coins_matrix.reshape(-1));
increasing_series = np.arange(1, 50)
plot_acf(increasing_series);
strong_autocorr_positive = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 ,1])
plot_acf(strong_autocorr_positive);
