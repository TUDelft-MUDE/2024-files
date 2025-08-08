# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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

# %% [markdown]
# **End of notebook.**
#
# <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
#   <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#   </div>
#   <div style="font-size: 75%; margin-top: 10px; text-align: right;">
#     By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
#     &copy; 2024 TU Delft. 
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
#     <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
