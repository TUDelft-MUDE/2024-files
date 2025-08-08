%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
from models import *
counts_2018 = np.genfromtxt('2018_count_250108_0209.csv', delimiter=',', skip_header=1)
day_min_counts = Minutes.day_min(counts_2018[:, 0])
print(f"Loaded {len(counts_2018)} rows of data")
print(f"The shape is {day_min_counts.shape}")
tickets_2018 = np.genfromtxt('2018_tickets_250108_0209.csv', delimiter=',', skip_header=1)
day_min_tickets = Minutes.day_min(tickets_2018)
print(f"Loaded {len(day_min_tickets)} rows of data")
print(f"The shape is {tickets_2018.shape}")
print()
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
radius_c = Minutes.radius(day_c, min_c)
day_a, min_a = transform(day_min_all[:,0], day_min_all[:,1])
radius_a = Minutes.radius(day_a, min_a)
radius_max = np.max(radius_a)
print(f"Maximum radius is: {radius_max:.3f}")
def get_band_distribution(minute,
                          radius_c, counts_2018, radius_a, transform,
                          std_increment=0.125):
    day_c, min_c = Minutes.get_day_min(minute[0])
    day_t, min_t = transform(day_c, min_c)
    radius = Minutes.radius(day_t, min_t)
    if isinstance(std_increment, list):
        assert len(std_increment) == 2, "std_increment list must have two values"
        radius_min = radius - std_increment[0]
        radius_max = radius + std_increment[1]
    elif isinstance(std_increment, float):
        radius_min = radius - std_increment
        radius_max = radius + std_increment
    else:
        raise ValueError("std_increment must be a list of length 2 or a float")
    if radius_min < 0:
        d_radius = radius_max - radius_min
        radius_min = 0
        radius_max = d_radius
    if radius_max > max(radius_a):
        d_radius = radius_max - radius_min
        radius_min = radius_min - d_radius
        radius_max = max(radius_a)
    d = evaluate_ticket_dist_i(radius_min, radius_max,
                             radius_c, counts_2018[:, 1],
                             radius_a)
    return d
def scale_minute(minute):
            """see colab notebook"""
            eta=[2,2.2,1.8,300,3,1.6,0.1,0.45]
            minute_in_hour = minute % 60  
            scaling_factor=1
            if minute_in_hour == 0:
                scaling_factor = eta[0]
            elif minute_in_hour==15:
                scaling_factor = eta[1]
            elif minute_in_hour==30:
                scaling_factor = eta[2]
            elif 1<minute<250: # correction for minutes just after midnight
                scaling_factor = eta[3]*1/(minute)
            elif minute==24*60:
                scaling_factor = eta[4]
            elif minute>24*60-5:# correction for minutes just before midnight
                scaling_factor = eta[5]
            else:
                scaling_factor = 1 - eta[6] * (minute_in_hour % 10) / 10 
                minute_scaling_factor = 1 - eta[7]*minute_in_hour / 60.0
                scaling_factor *= minute_scaling_factor 
            return scaling_factor
def shift_distribution(values, mode_original,minute):
    'shitfs the distribution to the right, making higher values more likely, pads with zeros the new vales on the left'
    mode_new = mode_original *scale_minute(minute)
    shift_amount = int(mode_new - mode_original)
    new_distribution = [0] * shift_amount
    if shift_amount > 0:
        new_distribution.extend(values)
    else:
        new_distribution = values
    return new_distribution
minute = Minutes.get_minutes([['April'], 29, 13, 30])
print(f"Minute: {minute[0]}")
d = get_band_distribution(minute,
                          radius_c, counts_2018, radius_a, transform,
                          std_increment=[.2, .2])
d.hist();
minutes = np.arange(0,60*1440).tolist()
dist_all_minutes = []
for minute in minutes:
    dist_all_minutes.append(get_band_distribution(
            [minute], radius_c, counts_2018,
            radius_a, transform, std_increment=0.125
            ))
list_of_distributions = []
for i,obj in enumerate(dist_all_minutes):
    if (obj.stats['mean']<0.5) and (obj.stats['mode']==0):
        kde =[1/obj.stat['mean'],1-(1/obj.stat['mean'])]
    else:
         kde=obj.kde
    print(i)
    kde=shift_distribution(kde, obj.stats['mode'], i)
    list_of_distributions.append(kde)
with h5py.File('data.h5', 'w') as h5file:
    for i, sublist in enumerate(list_of_distributions):
        h5file.create_dataset(f'list_{i}', data=sublist)
with open('pickles/List_of_kde_scaled.pkl', 'wb') as f:
    pickle.dump(list_of_distributions, f)
print("RadialDist list saved to 'List_of_radial_object_all_minutes.pkl'")
print()
