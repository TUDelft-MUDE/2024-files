%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
from models import *
counts_2018 = np.genfromtxt('2018_count_250108_0209.csv', delimiter=',', skip_header=1)
print(f"Loaded {len(counts_2018)} rows of data")
print(f"The shape is {counts_2018.shape}")
tickets_2018 = np.genfromtxt('2018_tickets_250108_0209.csv', delimiter=',', skip_header=1)
print(f"Loaded {len(tickets_2018)} rows of data")
print(f"The shape is {tickets_2018.shape}")
counts_2018[:5, :]
counts_array = Models.map_data_to_day_min(counts_2018[:,1], counts_2018[:,0])
m = Models()
m.plot(counts_array,
       custom_label="Count for each ticket",
       custom_title="Count for each ticket",
       custom_colors='Greys')
day_min_tickets = Minutes.day_min(tickets_2018)
day_min_counts = Minutes.day_min(counts_2018[:, 0])
print(day_min_tickets.shape)
print(day_min_counts.shape)
print(f"The mean day is {np.mean(day_min_tickets[:, 0])}")
print(f"The std day is {np.std(day_min_tickets[:, 0])}")
print(f"The mean min is {np.mean(day_min_tickets[:, 1])}")
print(f"The std min is {np.std(day_min_tickets[:, 1])}")
all = Tickets()
all.add([[1, 60]])
day_min_all = Minutes.day_min(all.tickets)
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
print(np.max(radius_a))
intervals = np.arange(0, 4.75, 0.25)
intervals = np.append(intervals, np.max(radius_a))
di = evaluate_ticket_dist_i(intervals[0], intervals[1],
                             radius, counts_2018[:, 1],
                             radius_a)
di.summarize_stats()
fit=di.hist()
print(fit)
d_all = evaluate_ticket_dist_all(intervals, radius, counts_2018[:, 1], radius_a)
print(d_all)
plt.plot(d_all[1].kde.pdf(np.arange(0,80,1)), label='All Tickets')
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
f = d_all[1].hist()
for i in range(len(d_all)):
    d_all[i].hist(include_zeros=True)
with open('pickles/List_of_radial_object.pkl', 'wb') as f:
    pickle.dump(d_all, f)
print("RadialDist list saved to 'radial_dist_list.pkl'")
