# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: mude-week-8
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from scipy import stats
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import matplotlib


# %%
data = np.array([[2.1, 2.6, 4.3, 3.8, 2.5, 4.7, 1.4, 1.9, 3.6, 3.1],
                 [5.1, 3.2, 7.2, 4.8, 6.5, 4.1, 2.4, 6.2, 6.9, 3.6]])


# %%

labels = ['Settlement, $D$ [m]',
          'Sea Level Rise, $S$ [m]',
          'Observations of Coral Atolls']

# plot data
plt.scatter(data[0], data[1])
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.title(labels[2])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(0, 11, 2))
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('coral_atolls_data.svg')
plt.show()

# %%
threshold = [4, 6]
mask = [(data[0]>threshold[0]) & (data[1]>threshold[1])]

plt.scatter(data[0], data[1], c='k', label='Data')
plt.scatter(data[0][mask[0]], data[1][mask[0]], c='r', marker='s', label='Exceedances')
plt.vlines(threshold[0], 0, 10, colors='k', linestyles='dashed')
plt.hlines(threshold[1], 0, 10, colors='k', linestyles='dashed')
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.title(labels[2])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(0, 11, 2))
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('coral_atolls_data_threshold.svg')
plt.show()

# %%
x_part = data[0] - data[0].mean()
y_part = data[1] - data[1].mean()
x_y_parts = x_part*y_part
print(x_part, '\n', y_part)
print(x_y_parts)
cov = np.sum(x_y_parts)/len(x_part)
print(cov)
print(np.cov(data[0], data[1]))
print(cov*10/9)

# %%
print(data.mean(axis=1))
print(data.std(axis=1))
r = np.corrcoef(data)
print(r)
cov = np.cov(data)
print(cov)
print(stats.pearsonr(data[0], data[1])[0])

# %%
D = data[0]
S = data[1]
Z = 10 - D - S
print(Z.mean())
print(Z.std())

# %%
print(2/1.9)
stats.norm.cdf(0, loc=2, scale=1.9)

# %%
threshold = 0
mask = [Z<threshold]
print(mask)
plt.scatter(data[0], data[1], c='k', label='Data')
plt.scatter(data[0][mask[0]], data[1][mask[0]], c='r', marker='s', label='Exceedances')
plt.plot([10, 0], [0, 10], 'k--', label='Z = 0')
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.title(labels[2])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(0, 11, 2))
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('coral_atolls_data_function.svg')
plt.show()

# %%
