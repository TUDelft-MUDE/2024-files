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
# %load_ext autoreload
# %autoreload 2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from minutes import *

import random
import matplotlib.cm as cm

with open('pickles\List_of_kde_scaled.pkl', 'rb') as f:
    loaded_radial_dist_list = pickle.load(f)




# %%
loaded_radial_dist_list[0]
loaded_radial_dist_list[32484][20]

# %%
print(len(loaded_radial_dist_list))

# for list in loaded_radial_dist_list:
#     print(len(list))

loaded_list_lengths = [len(list) for list in loaded_radial_dist_list]
loaded_list_sum = [np.array(array).sum() for array in loaded_radial_dist_list]
loaded_list_sum = np.array(loaded_list_sum)
print(f"The maximum length of the lists is {max(loaded_list_lengths)}")
print(f"{sum(loaded_list_sum[np.isclose(loaded_list_sum,1.0,rtol=0.001)])} lists have a sum of 1.0")

loaded_list_zeros = [array[0]==0 for array in loaded_radial_dist_list]
print(f"{sum(loaded_list_zeros)} lists have a zero value at the first index")
print(f" (this is {sum(loaded_list_zeros)/len(loaded_list_zeros)*100:.2f}% of the lists)")

fig, ax1 = plt.subplots(figsize=(15, 5))

color = 'tab:blue'
ax1.set_xlabel('List Length')
ax1.set_ylabel('Number of Lists', color=color)
ax1.hist(loaded_list_lengths, bins=50, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Distribution', color=color)
ax2.hist(loaded_list_lengths, bins=50, cumulative=True, color=color, alpha=0.6)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Histogram and Cumulative Distribution of List Lengths')
plt.show()


# %%
len(loaded_radial_dist_list)

# %%
n_plots = 20
random_indices = random.sample(range(len(loaded_radial_dist_list)), n_plots)
biggest_indices = sorted(range(len(loaded_radial_dist_list)), key=lambda i: len(loaded_radial_dist_list[i]), reverse=True)[:n_plots]

plot_indices = [random_indices, biggest_indices]

plt.figure(figsize=(15, 5))
for idx in plot_indices[0]:
    rd = loaded_radial_dist_list[idx]
    plt.plot(rd, label=f'minute: {idx}')  
 



plt.legend()
plt.title('Random Radial Distribution with Indices')
plt.xlabel('Count')
plt.ylabel('density')
plt.show()

# %%
from models import sample_integer

def plot_sample_distribution(sample, probabilities):
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=np.arange(len(probabilities) + 1), density=True, alpha=0.6, color='g', label='Sampled Density')

    # Verification plot
    plt.plot(np.arange(len(probabilities)), probabilities, 'ro-', label='True Probabilities')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram of Sampled Values with True Probabilities')
    plt.legend()
    plt.show()

minute = 5*1440 + 60*13 + 1
print(minute)

# Example usage
probabilities = loaded_radial_dist_list[minute]
sample_size = 1000
sample = sample_integer(probabilities, sample_size)

plot_sample_distribution(sample, probabilities)


# %%
from models import sample_ticket

ticket_sample_criteria = [0.05, 10]
probabilities = loaded_radial_dist_list[0]#[32484]
sample = sample_ticket(probabilities, *ticket_sample_criteria, verbose=True)
plot_sample_distribution(sample, probabilities)
print(probabilities)


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
