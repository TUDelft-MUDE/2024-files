import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
reference_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
def minutes_to_date(minutes):
    '''From minute, print the day and time.
    Indicates month, day, hour and time given a minute difference
    with respect to the 1st April.
    Does the opposite of date_to_minutes()
    Convention: minute is the str/int and minutes is the float.
    - Input: float with difference in minutes
    - Output: print corresponding month, day, hour and minute.
    '''
    new_date = reference_date + timedelta(minutes=minutes)
    month_name = calendar.month_name[new_date.month]
    return print(f"{minutes:.1f} minutes corresponds to:",
                 f"{month_name} {new_date.day}",
                 f"at {new_date.hour}:{new_date.minute}.")
def date_to_minutes(month, day, hour, minute, return_float=False):
    '''From the day and time (str or int), return minutes (float).
    Indicates minute difference wrt 1st April given 4 strings
    that specify with month, day, hour, minute.
    Does the opposite of minutes_to_date()
    Convention: minute is the str/int and minutes is the float.
    - Input: 4x str or int: month, day, hour, minute
    - Output: float of minutes of difference wrt 1st April.
    '''
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    new_date = datetime(year=2024, month=month, day=day, hour=hour, minute=minute)
    minutes = (new_date - reference_date).total_seconds() / 60
    if return_float:
        return minutes
    else:
        return print(f"Month {month} day {day} at time {hour}:{minute}",
                     f" corresponds to: {minutes} minutes.")
def minutes_to_time(minutes):
    '''Print hours and minutes in a day given total minutes (float).
    Does the opposite of time_to_minutes()
    Convention: minute is the str/int and minutes is the float.
    - Input: float of minutes.
    - Output: print with corresponding hour and minutes.
    '''
    hour = minutes // 60
    minute = minutes % 60
    return print(f"{minutes:.1f} minutes corresponds to time",
                 f"{hour}:{minute}.")
def time_to_minutes(hour, minute, return_float=False):
    '''Prints minutes in a day given hour and minute (str or int).
    Does the opposite of minutes_to_time()
    Convention: minute is the str/int and minutes is the float.
    - Input: str or int of hour and minute.
    - Output: print with corresponding minutes.
    '''
    hour = int(hour)
    minute = int(minute)
    minutes = hour*60 + minute
    if return_float:
        return minutes
    else:
        return print(f"Time {hour}:{minute}"
                     f" corresponds to: {minutes} minutes.")
def prob_between_floats(time_float_start, time_float_end, mu, std):
    '''
    Calculate probability given two floats and the mean and
    standard deviation of a Gaussian fit.
    Can handle any of the 3 types of time units, as long as 
    arguments (all floats) are used consistently.
    - Input:
       - time_float_start: starting float in time
       - time_float_end: end minute
       - mu: mean of the normal fit
       - std: standard deviation of the normal fit
    - Output: 
       - Probability of a event happening between those minutes
    '''
    cdf_start = norm.cdf(time_float_start, mu, std)
    cdf_end = norm.cdf(time_float_end, mu, std)
    return cdf_end - cdf_start
ice_data = pd.read_csv('data.csv')
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
ice_data
date_to_minutes('4', '15', '10', '30')
date_to_minutes(4, 15, 10, 30)
date_to_minutes('4', '15', '10', '30', return_float=True)
minutes_to_date(2000)
date_to_minutes(4, 1, 0, 0)
minutes_to_date(date_to_minutes(4, 1, 0, 0, return_float=True))
minutes_to_date(date_to_minutes(5, 1, 0, 0, return_float=True))
data = ice_data['minutes']
mu_min, std_min = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_min, std_min)
plt.figure(1)
plt.hist(data, bins=30, density=True, alpha=0.8, color='g',
        stacked=True,  edgecolor='black', linewidth=1.2)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_min, std_min)
plt.title(title)
plt.xlabel('Minutes since reference date')
plt.ylabel('Histogram/PDF')
plt.show()
test_min_1 = 26670
test_min_2 = 26671
print(f'Function prob_between_floats returns type: ',
      f'{type(prob_between_floats(test_min_1, test_min_2, mu_min, std_min))}')
print(f'Probability between {test_min_1} min and {test_min_2} min',
      f'is {prob_between_floats(test_min_1, test_min_2, mu_min, std_min):.3e}')
prob_between_floats(test_min_1, test_min_2, mu_min, std_min)
minutes_to_time(875)
time_to_minutes('14', '35')
time_to_minutes(14, 35)
data = ice_data['minutes_in_day']
mu_time, std_time = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_time, std_time)
plt.figure(1)
plt.hist(data, bins=20, density=True, alpha=0.8, color='g',
        stacked=True,  edgecolor='black', linewidth=1.2)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(x = 720, color = 'black', linestyle = ':', alpha = 0.5)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_time, std_time)
plt.title(title)
plt.xlabel('Minutes during day (noon = 720 min)')
plt.ylabel('Histogram/PDF')
plt.show()
prob_between_floats(875, 876, mu_time, std_time)
days = (ice_data['datetime'] - ice_data['ref_date_annual']).dt.total_seconds()/(60*60*24)
data = days
mu_time_as_day, std_time_as_day = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_time_as_day, std_time_as_day)
plt.figure(1)
plt.hist(data, bins=30, density=True, alpha=0.8, color='g',
        stacked=True,  edgecolor='black', linewidth=1.2)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(x = 32, color = 'black', linestyle = ':', alpha = 0.5)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_time_as_day, std_time_as_day)
plt.title(title)
plt.xlabel('Days since reference_date (May 1 = 32 days)')
plt.ylabel('Histogram/PDF')
plt.show()
prob_between_floats(31, 32, mu_time_as_day, std_time_as_day)
my_month = 4
my_day = 29
my_hour = 12
my_min = 30
prediction_min = date_to_minutes(my_month, my_day, my_hour, my_min,
                                 return_float=True)
prediction_min_in_day = time_to_minutes(my_hour, my_min,
                                        return_float=True)
date_to_minutes(my_month, my_day, my_hour, my_min)
time_to_minutes(my_hour, my_min)
p1 = prob_between_floats(prediction_min,
                         prediction_min+1,
                         mu_min, std_min)
print(f'Probability of my guess is: {p1:.3e}')
p2 = prob_between_floats(prediction_min_in_day,
                         prediction_min_in_day+1,
                         mu_time, std_time)
print(f'Probability of my guess is: {p2:.3e}')
p3 = prob_between_floats(my_day,
                         my_day+1,
                         mu_time_as_day,
                         std_time_as_day)
print(f'Probability of my guess is: {p3:.3e}')
p_prediction = p2*p3
print(f'Probability of my prediction is: {p_prediction:.3e}')
