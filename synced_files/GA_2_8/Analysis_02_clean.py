%load_ext autoreload
%autoreload 2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tools import *
pickle_path = os.path.join('pickles',
                           'tickets_per_minute.pkl')
with open(pickle_path, 'rb') as f:
    loaded_radial_dist_list = pickle.load(f)
def plot_sample_distribution(sample, probabilities):
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=np.arange(len(probabilities) + 1), density=True, alpha=0.6, color='g', label='Sampled Density')
    plt.plot(np.arange(len(probabilities)), probabilities, 'ro-', label='True Probabilities')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Number of Tickets Purchased for Specific Minute')
    plt.legend()
    plt.show()
trial_day = 23
trial_min_in_day = 13*60 + 30
trial_min = trial_day*24*60 + trial_min_in_day
probabilities = loaded_radial_dist_list[trial_min]
sample_size = 1000
sample = sample_integer(probabilities, sample_size)
plot_sample_distribution(sample, probabilities)
trial_day = 20
trial_min_in_day = 14*60 + 15
trial_min = trial_day*24*60 + trial_min_in_day
probabilities = loaded_radial_dist_list[trial_min]
sample_size = 1000
sample = sample_integer(probabilities, sample_size)
expected_value_N_w = int(round(np.mean(sample)))
m = Models()
p_ticket = m.get_p([trial_min])
expected_value_profit = p_ticket*300000/expected_value_N_w - 3
print(f"The expected number of tickets purchased is {expected_value_N_w}")
print(f"The expected profit is {expected_value_profit:.2f} USD")
def plot_hist_and_cdf(sampled_profit, xlim=None, ylim=None):
    fig, ax1 = plt.subplots()
    ax1.hist(sampled_profit, bins=50, edgecolor='k', density=True, alpha=0.6, label='Histogram')
    ax1.set_xlabel('Winnings, thousands of USD')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    sorted_samples = np.sort(sampled_profit)
    cdf = np.arange(1, len(sorted_samples) + 1) / (len(sorted_samples) + 1)
    ax2.plot(sorted_samples, cdf, color='red', label='CDF')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc='upper right')
    plt.show()
    fig, ax = plt.subplots()
    exceedance_prob = 1 - cdf
    ax.plot(sorted_samples, exceedance_prob, color='blue', label='Exceedance Probability')
    ax.vlines(300, 0, 1, color='grey', linestyle='--', label='300k threshold')
    ax.set_xlabel('Winnings, thousands of USD')
    ax.set_ylabel('Exceedance Probability')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right')
    plt.show()
t = Tickets()
t.add([[27]])
t.status()
t.show()
m = Models()
prob_T = np.zeros((len(t.tickets)))
Nw_T = np.zeros((len(t.tickets)))
for i, ti in enumerate(t.tickets):
    prob_T[i] = m.get_p([ti])
    day, min = Minutes.get_day_min(ti)
    Nw_T [i] = m.ticket_model(day, min)
prob_T_matrix = m.map_data_to_day_min(prob_T, t.tickets)
prob_all = np.sum(prob_T)
print(f"The probability of any ticket winning is {prob_all:.3e}")
cost_tickets = len(t.tickets)*3
expected_winnings = np.sum(300*prob_T/(Nw_T + 1))
expected_profit = expected_winnings - .003*len(t.tickets)
print(f"Number of tickets: \t{len(t.tickets)}")
print(f"Cost tickets: \t\t{cost_tickets/1000:9.2e} kUSD \t({len(t.tickets)*3:5.0f} USD)")
print(f"Expected winnings: \t{expected_winnings:9.2e} kUSD \t({expected_winnings*1000:5.0f} USD)")
print(f"Expected profit: \t{expected_profit:9.2e} kUSD \t({expected_profit*1000:5.0f} USD)")
m.plot(prob_T_matrix)
Ns = 100000
sampled_ticket = np.zeros((Ns,), dtype=int)
sampled_probability = np.zeros((Ns,))
sampled_winnings = np.zeros((Ns,))
for i in range(Ns):
    sampled_ticket[i] = random.choices(t.tickets, weights=prob_T, k=1)[0]
    sampled_probability[i] = prob_T[
        t.tickets.index(sampled_ticket[i])]
    sampled_winnings[i] = 300/(1 + sample_integer(
        loaded_radial_dist_list[sampled_ticket[i]]))
plot_hist_and_cdf(sampled_winnings,
                  xlim=(1,500), ylim=(1e-2, 1))
print("=====================================")
print("    USING EXPECTED VALUE CALC    ")
print("=====================================")
print(f"Number of tickets: \t{len(t.tickets)}")
print(f"Cost tickets: \t\t{cost_tickets/1000:9.2e} kUSD \t({len(t.tickets)*3:5.0f} USD)")
print(f"Expected winnings: \t{expected_winnings:9.2e} kUSD \t({expected_winnings*1000:5.0f} USD)")
print(f"Expected profit: \t{expected_profit:9.2e} kUSD \t({expected_profit*1000:5.0f} USD)")
