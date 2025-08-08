import numpy as np
import matplotlib.pyplot as plt
C = YOUR_CODE_HERE
alpha = YOUR_CODE_HERE
pf_societal = YOUR_CODE_HERE
print(pf_societal)
N_values = YOUR_CODE_HERE
limit_line = YOUR_CODE_HERE
fig, ax = plt.subplots(figsize=(8, 6))
YOUR_CODE_HERE
YOUR_CODE_HERE
ax.set_title('YOUR_CODE_HERE')
ax.set_xlabel('YOUR_CODE_HERE')
ax.set_ylabel('YOUR_CODE_HERE')
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
ax.grid(True)
plt.show()
n_and_p = np.array([[1,   0.099],
                    [2,   7e-4],
                    [5,   2e-4],
                    [10,  9e-5],
                    [20,  5e-6],
                    [60,  3e-6],
                    [80,  6e-7],
                    [100,  4e-7],
                    [150, 8e-8]])
N_plot = n_and_p[:, 0]
p_cumulative = YOUR_CODE_HERE
fig, ax = plt.subplots(figsize=(8, 6))
YOUR_CODE_HERE
YOUR_CODE_HERE
YOUR_CODE_HERE
ax.set_title('YOUR_CODE_HERE')
ax.set_xlabel('YOUR_CODE_HERE')
ax.set_ylabel('YOUR_CODE_HERE')
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
ax.grid(True)
plt.show()
n_and_p_modified = n_and_p.copy()
n_and_p_modified[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE
DUPLICATE_ANALYSIS_FROM_ABOVE_WITH_MODIFIED_VALUES
YOUR_CODE_HERE
