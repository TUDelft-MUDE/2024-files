import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# Define bin edges and counts
bin_edges = np.array([0, 1, 2, 3, 6, 55])
bin_counts = np.array([5, 15, 25, 35, 20])

# Plot histogram with counts
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(bin_edges[:-1], bin_counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.title('Histogram with Counts')

# Convert counts to density
bin_widths = np.diff(bin_edges)
bin_density = bin_counts / (np.sum(bin_counts))

# Plot histogram with density
plt.subplot(1, 2, 2)
plt.bar(bin_edges[:-1], bin_density, width=bin_widths, edgecolor='black', align='edge')
plt.xlabel('Bins')
plt.ylabel('Density')
plt.title('Histogram with Density')

plt.tight_layout()
plt.show()

print(sum(bin_density))