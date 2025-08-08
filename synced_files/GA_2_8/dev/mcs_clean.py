%load_ext autoreload
%autoreload 2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tickets import *
from minutes import *
from models import *
pickle_path = os.path.join('pickles',
                           'List_of_kde_scaled.pkl')
with open(pickle_path, 'rb') as f:
    loaded_radial_dist_list = pickle.load(f)
def sample_benefit_test(probabilities, cov, N_min, N_max, verbose=True):
    sample = sample_ticket(probabilities, cov, N_min, N_max, verbose)
    return 300/(sample + 1)
prob = loaded_radial_dist_list[0]#[32484]
sample = sample_benefit_test(prob, 0.1, 100, 10000)
unique, counts = np.unique(sample, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value: {value}, Frequency: {count}")
plt.hist(sample, bins=50, edgecolor='k')
plt.xlabel('Benefit')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Benefits')
plt.show()
def plot_hist_and_cdf(sampled_profit, xlim=None, ylim=None):
    fig, ax1 = plt.subplots()
    ax1.hist(sampled_profit, bins=50, edgecolor='k', density=True, alpha=0.6, label='Histogram')
    ax1.set_xlabel('thousands of dollars')
    ax1.set_ylabel('Frequency')
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
t = Tickets()
t.add([[27, 28, 29]])
t.status()
t.show()
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
    sampled_winnings[i] = 300/(1 +sample_integer(
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
print("=====================================")
print("    USING MONTE CARLO SIMULATION    ")
print("=====================================")
print(f"Tickets purchased: {t.N()}")
FD_expected_winnings = np.sum(sampled_probability*sampled_winnings/(Ns+1))
print(f"Expected winnings: {FD_expected_winnings:9.2e} kUSD ({FD_expected_winnings*1000:5.0f} USD)")
FD_expected_profit = FD_expected_winnings - t.N()/1000*3
print(f"Expected profit:   {FD_expected_profit:9.2e} kUSD ({FD_expected_profit*1000:5.0f} USD)")
sampled_loss = 3*t.N()/1000 - sampled_winnings
print(f"Min loss =    {np.min(sampled_loss):8.3f}")
print(f"Max loss =    {np.max(sampled_loss):8.3f}")
print(f"Mean loss =   {np.mean(sampled_loss):8.3f}")
print(f"Median loss = {np.median(sampled_loss):8.3f}")
print(f"Std loss =    {np.std(sampled_loss):8.3f}")
plot_hist_and_cdf(sampled_loss, xlim=(.001,100))
sorted_loss = np.sort(sampled_loss)[::-1]
exceedance_prob = np.arange(1, len(sorted_loss) + 1) / (len(sorted_loss) + 1)
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
type(int(round(np.random.choice(t.tickets))))
test = np.zeros((3,))
print(test)
print(test.shape)
test = np.append(test, [1,2,3])
test.shape
print(test)
