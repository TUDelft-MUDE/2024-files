# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %% [markdown]
# # MCS
#
# _see markdown file for half-baked theory_

# %% [markdown]
# Constants:
# - $V_n$: coefficient of variation of the expected value of the winning ticket
# - $N_{s,min}$: minimum number of samples to estimate the expected value of the number of winning tickets
#
# Procedure is:
#
# 1. Determine the set of tickets (size $N_t$) to purchase, $T$, defined by minutes $m$
# 2. Compute total cost $3\cdot N_t$
#
# For each ticket $i$ in $T$ associated with minute $m_i$:
# 1. Compute probability of being the winning ticket, $p_i=P[M=m_i=m_w]$
# 2. Sample $N_w(m_i)$; stop when expected value of samples satisfies the criteria defined above (sample index $j=1,\ldots,N_{s,i}$)
# 3. Compute the profit from each realization $j$ for minute $i$ as $W \cdot p_i / N_{w, j}$
#
# The samples of $B$ are the collection of all profits computed for the samples of $i$

# %%
# %load_ext autoreload
# %autoreload 2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tickets import *
from minutes import *
from models import *

# %%
pickle_path = os.path.join('pickles',
                           'List_of_radial_object_all_minutes_v3.pkl')

with open(pickle_path, 'rb') as f:
    loaded_radial_dist_list = pickle.load(f)


# %%

# %%
def sample_benefit_test(probabilities, cov, N_min, N_max, verbose=True):
    sample = sample_ticket(probabilities, cov, N_min, N_max, verbose)
    return 300/(sample + 1)



prob = loaded_radial_dist_list[0]#[32484]
sample = sample_benefit_test(prob, 0.1, 100, 10000)


# %%
unique, counts = np.unique(sample, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value: {value}, Frequency: {count}")

plt.hist(sample, bins=50, edgecolor='k')
plt.xlabel('Benefit')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Benefits')
plt.show()


# %%


def plot_hist_and_cdf(sampled_profit, xlim=None, ylim=None):
    fig, ax1 = plt.subplots()

    # Plot histogram
    ax1.hist(sampled_profit, bins=50, edgecolor='k', density=True, alpha=0.6, label='Histogram')
    ax1.set_xlabel('thousands of dollars')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the cumulative distribution
    ax2 = ax1.twinx()
    sorted_samples = np.sort(sampled_profit)
    cdf = np.arange(1, len(sorted_samples) + 1) / (len(sorted_samples) + 1)
    ax2.plot(sorted_samples, cdf, color='red', label='CDF')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc='upper right')

    # plt.title('Histogram and Cumulative Distribution of Sample Benefits')
    plt.show()

    # Plot exceedance probability
    fig, ax = plt.subplots()
    exceedance_prob = 1 - cdf
    ax.plot(sorted_samples, exceedance_prob, color='blue', label='Exceedance Probability')
    ax.vlines(300, 0, 1, color='grey', linestyle='--', label='300k threshold')
    ax.set_xlabel('thousands of dollars')
    ax.set_ylabel('Exceedance Probability')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right')
    plt.show()



# %%
t = Tickets()
t.add([[27, 28, 29], [13]])
# t.add([[14], [13]])
t.status()
t.show()

# %%
m = Models(model_id=2)
prob_T = np.zeros((len(t.tickets)))
Nw_T = np.zeros((len(t.tickets)))
for i, ti in enumerate(t.tickets):
    prob_T[i] = m.get_p([ti])
    day, min = Minutes.get_day_min(ti)
    Nw_T [i] = m.ticket_model(day, min)
prob_T_matrix = m.map_data_to_day_min(prob_T, t.tickets)
prob_all = np.sum(prob_T)
print(f"The probability of a ticket winning is {prob_all:.3e}")
cost_tickets = len(t.tickets)*3
expected_winnings = np.sum(300*prob_T/Nw_T)
expected_profit = expected_winnings - .003*len(t.tickets)
print(f"Number of tickets: \t{len(t.tickets)}")
print(f"Cost tickets: \t\t{cost_tickets/1000:9.2e} kUSD \t({len(t.tickets)*3:5.0f} USD)")
print(f"Expected winnings: \t{expected_winnings:9.2e} kUSD \t({expected_winnings*1000:5.0f} USD)")
print(f"Expected profit: \t{expected_profit:9.2e} kUSD \t({expected_profit*1000:5.0f} USD)")

m.plot(prob_T_matrix)

# %%
Ns = 100000
sampled_ticket = np.zeros((Ns,), dtype=int)
sampled_profit = np.zeros((Ns,))
for i in range(Ns):
    sampled_ticket[i] = random.choices(t.tickets, weights=prob_T, k=1)[0]
    sampled_profit[i] = 300/(1 +sample_integer(
        loaded_radial_dist_list[sampled_ticket[i]]))




# %% [markdown]
# Running the simulation for N simulations takes:
#
# | N | Time |
# | :---: | :---: |
# | 1e7 | stopped after 5 min |
# | 1e5 | 10 s |
#
#

# %%
plot_hist_and_cdf(sampled_profit,
                  xlim=(1,500), ylim=(1e-2, 1))

# %%
print(t.N())
total_loss = 3*t.N()/1000 - sampled_profit
print(f"Ticket cost = {t.N()/1000} kUSD ({t.N()} USD))")
print(f"Min loss = {np.min(total_loss)}")
print(f"Max loss = {np.max(total_loss)}")
print(f"Mean loss = {np.mean(total_loss)}")
print(f"Median loss = {np.median(total_loss)}")
print(f"Std loss = {np.std(total_loss)}")

plot_hist_and_cdf(total_loss)
# prob_T_matrix = m.map_data_to_day_min(prob_T, t.tickets)
# m.plot(prob_T_matrix)

# %%
# Sort the total_profit array in descending order
sorted_loss = np.sort(total_loss)[::-1]

# Calculate the exceedance probability
exceedance_prob = np.arange(1, len(sorted_loss) + 1) / (len(sorted_loss) + 1)

# Plot the exceedance probability
plt.figure(figsize=(10, 6))
plt.plot(sorted_loss, exceedance_prob, marker='o', linestyle='-', color='b')
plt.xlabel('Total Loss')
plt.ylabel('Exceedance Probability')
plt.title('Exceedance Probability Plot of Total Loss')
plt.xscale('log')
plt.yscale('log')
plt.xlim([10, 1000])
plt.ylim([0.1, 1])
plt.grid(True)
plt.show()

# %%
type(int(round(np.random.choice(t.tickets))))

# %%
test = np.zeros((3,))
print(test)
print(test.shape)
test = np.append(test, [1,2,3])
test.shape
print(test)

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
