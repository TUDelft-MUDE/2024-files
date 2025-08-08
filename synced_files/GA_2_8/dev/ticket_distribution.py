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
# # Distribution of Tickets Purchased per Minute
#
# We have all bets from 2018, which gives us an idea of the number of tickets purchased for any minute of the day. See results elsewhere; the purpose of this notebook is to identify a suitable distribution for the number of tickets purchased per minute. It should satisfy these criteria:
#
# - easily produce discrete (integer) values >0
# - have an expected value provided by the (empirical model of) 2018 data, which is formulated as a continuous function that is then rounded to the nearest integer
# - statistics are captured well
#
# |  | Payout | No. Winners |
# | :--: | :--: | :--: | 
# | mean	  |  155960	 |10,3 |
# | median  |	112500   | 7,0 |
# | mode	  |  80000    | 1,0 |
# | std	  |  97807    |10,8 |
# | min	  |  800      | 1,0 |
# | max	  |  363627   |58,0 |
#

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
from models import *

# %%
counts_2018 = np.genfromtxt('2018_count_250108_0209.csv', delimiter=',', skip_header=1)

print(f"Loaded {len(counts_2018)} rows of data")
print(f"The shape is {counts_2018.shape}")

tickets_2018 = np.genfromtxt('2018_tickets_250108_0209.csv', delimiter=',', skip_header=1)

print(f"Loaded {len(tickets_2018)} rows of data")
print(f"The shape is {tickets_2018.shape}")

# %%
counts_2018[:5, :]

# %%
counts_array = Models.map_data_to_day_min(counts_2018[:,1], counts_2018[:,0])

m = Models()
m.plot(counts_array,
       custom_label="Count for each ticket",
       custom_title="Count for each ticket",
       custom_colors='Greys')


# %%
day_min_tickets = Minutes.day_min(tickets_2018)
day_min_counts = Minutes.day_min(counts_2018[:, 0])

print(day_min_tickets.shape)
print(day_min_counts.shape)
print(f"The mean day is {np.mean(day_min_tickets[:, 0])}")
print(f"The std day is {np.std(day_min_tickets[:, 0])}")
print(f"The mean min is {np.mean(day_min_tickets[:, 1])}")
print(f"The std min is {np.std(day_min_tickets[:, 1])}")

# # Define the bin edges for x and y
# x_bins = np.arange(10, 50, 1)
# y_bins = np.arange(400, 1441, 60)

# # Create the 2D histogram
# plt.hist2d(day_min[:, 0], day_min[:, 1], bins=[x_bins, y_bins], cmap='viridis')

# # Add color bar
# plt.colorbar(label='Counts')

# # Add labels and title
# plt.xlabel('Column 0')
# plt.ylabel('Column 1')
# plt.title('2D Density Map')

# # Show the plot
# plt.show()

# %%
all = Tickets()
all.add([[1, 60]])
day_min_all = Minutes.day_min(all.tickets)

# %%
parameters = [np.mean(day_min_tickets[:, 0]),
              np.std(day_min_tickets[:, 0]),
              np.mean(day_min_tickets[:, 1]),
              np.std(day_min_tickets[:, 1])]

transform = Minutes.get_transform(parameters)

day_c, min_c = transform(day_min_counts[:,0], day_min_counts[:,1])
radius = Minutes.radius(day_c, min_c)

day_a, min_a = transform(day_min_all[:,0], day_min_all[:,1])
radius_a = Minutes.radius(day_a, min_a)

print(day_a)

m = Models(model_id=1)
m.plot(Models.map_data_to_day_min(radius, counts_2018[:, 0]),
       custom_label="Number of Std Devs from Mean",
       custom_title="Counts",
       custom_colors='summer')
m.plot(Models.map_data_to_day_min(radius_a, all.tickets),
       custom_label="Number of Std Devs from Mean",
       custom_title="Counts",
       custom_colors='summer')


# %%
print(np.max(radius_a))

# %%
intervals = np.arange(0, 4.75, 0.25)
intervals = np.append(intervals, np.max(radius_a))

di = evaluate_ticket_dist_i(intervals[0], intervals[1],
                             radius, counts_2018[:, 1],
                             radius_a)

di.summarize_stats()
fit=di.hist()
print(fit)

# %%
d_all = evaluate_ticket_dist_all(intervals, radius, counts_2018[:, 1], radius_a)

print(d_all)
# Testing
plt.plot(d_all[1].kde.pdf(np.arange(0,80,1)), label='All Tickets')

# %%
print('No.\tstd0\tstd1\tChosen\tTotal\tUnchsn\t%unch\ttotal')
for i, d in enumerate(d_all):
    print(f"{i}"
          +f"\t{d.range[0]:.2f}"
          +f"\t{d.range[1]:.2f}"
          +f"\t{d.N_chosen}"
          +f"\t{d.N_total}"
          +f"\t{d.N_unchosen}"
          +f"\t{(1-(d.N_chosen/d.N_total))*100:.2f}"
          +f"\t{int(np.sum(d.counts))}")

# %%
from matplotlib.pylab import f


print('No.\tstd1\t%unch\tmode\tN_mode\t%_mode\tmedian\tmean\tstd\tmin\tmax')
for i, d in enumerate(d_all):
    print(f"{i}"
          +f"\t{d.range[1]:.2f}"
          +f"\t{d.N_chosen/d.N_total*100:.2f}"
          +f"\t{d.stats['mode']:.2f}"
          +f"\t{d.stats['mode_count']}"
          +f"\t{d.stats['mode_count']/d.N_chosen*100:.0f}"
          +f"\t{d.stats['median']:.2f}"
          +f"\t{d.stats['mean']:.2f}"
          +f"\t{d.stats['std']:.2f}"
          +f"\t{d.stats['min']:.2f}"
          +f"\t{d.stats['max']:.2f}")
          

# %%
f = d_all[1].hist()

# %%
for i in range(len(d_all)):
    d_all[i].hist(include_zeros=True)

# %%
with open('pickles/List_of_radial_object.pkl', 'wb') as f:
    pickle.dump(d_all, f)

print("RadialDist list saved to 'radial_dist_list.pkl'")

# %% [markdown]
#

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
