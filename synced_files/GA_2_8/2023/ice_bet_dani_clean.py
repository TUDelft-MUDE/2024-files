import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
def minute_diff_to_date(minute_diff):
    '''
    Indicates month, day, hour and time given a minute difference with respect to the 1st April.
    - Input: float with difference in minutes
    - Output: print of the corresponding month, day, hour and minute.
    '''
    specific_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
    new_date = specific_date + timedelta(minutes=minute_diff)
    month_name = calendar.month_name[new_date.month]
    return print(f"The minute difference corresponds to: Month {month_name}, Day {new_date.day}, Hour {new_date.hour}, Minute {new_date.minute}")
def time_to_minute_diff(month, day, hour, minute):
    '''
    Indicates minute difference wrt 1st April given 4 strings with month, day, hour, minute.
    - Input: separate string for month, day, hour and minute.
    - Output: float of minutes of difference wrt 1st April.
    '''
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    specific_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
    new_date = datetime(year=2024, month=month, day=day, hour=hour, minute=minute)
    minute_diff = (new_date - specific_date).total_seconds() / 60
    return print(f"The minute difference is {minute_diff}")
def minutes_to_time(minutes):
    '''
    Indicates hours and minutes in a day given total minutes.
    - Input: float of minutes.
    - Output: print with corresponding hour and minutes.
    '''
    hour = minutes // 60
    minute = minutes % 60
    return print(f"The minutes correspond to: Hour {hour}, Minute {minute}")
def time_to_minutes(hour, minute):
    '''
    Indicates minutes in a day given hours and minutes.
    - Input: strings of hour and minute.
    - Output: print with corresponding minutes.
    '''
    hour = int(hour)
    minute = int(minute)
    minutes = hour * 60 + minute
    return print(f"The number of minutes passed in the day is {minutes}")
def prob_between_times(time_float_start, time_float_end, mu, std):
    '''
    Calculate probability given two floats and the mean and standard deviation of a Gaussian fit.
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
specific_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
keep_cols = [0,1,2,7,4]
skip_rows = 0
ice_data = pd.read_excel('./data.xlsx', sheet_name = 'Worksheet', skiprows=skip_rows, usecols=keep_cols)
ice_data['Hour'] = np.floor(ice_data['Hour (24)'])
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['minutes_diff'] = np.abs((ice_data['datetime'] - specific_date).dt.total_seconds() / 60)
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['Year'] = ice_data['datetime'].dt.year
ice_data['Month'] = ice_data['datetime'].dt.month
ice_data['Day'] = ice_data['datetime'].dt.day
ice_data['Hour'] = ice_data['datetime'].dt.hour
ice_data['Minute'] = ice_data['datetime'].dt.minute
ice_data['reference_date'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
ice_data['minutes_diff'] = (ice_data['datetime'] - ice_data['reference_date']).dt.total_seconds() / 60
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute
ice_data
time_to_minute_diff('4', '15', '10', '30')
minute_diff_to_date(2000)
data = ice_data['minutes_diff']
mu_date, std_date = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_date, std_date)
plt.figure(1)
plt.hist(data, bins=30, density=True, alpha=0.8, color='g')
plt.figure(2)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_date, std_date)
plt.title(title)
plt.show()
prob_between_times(26670, 26671, mu_date, std_date)
minutes_to_time(875)
time_to_minutes('14', '35')
data = ice_data['minutes_in_day']
mu_day, std_day = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_day, std_day)
plt.figure(1)
plt.hist(data, bins=20, density=True, alpha=0.8, color='g')
plt.figure(2)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_day, std_day)
plt.title(title)
plt.show()
prob_between_times(875, 876, mu_day, std_day)
day_diff = (ice_data['datetime'] - ice_data['reference_date']).dt.total_seconds() / (60*60*24)
data = day_diff
mu_2, std_2 = norm.fit(data)
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_2, std_2)
plt.figure(1)
plt.hist(data, bins=30, density=True, alpha=0.8, color='g')
plt.figure(2)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu_2, std_2)
plt.title(title)
plt.show()
prob_between_times(31, 32, mu_2, std_2)
