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
# # Project 12: Evidence-Based Gambling
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.8. Friday January 18, 2024.*

# %% [markdown]
# ## Introduction
#
# In this exercise we evaluate the data from previous Ice Classic years to check the quality of our break-up prediction.
#
# To avoid becoming mindless [gamblers](https://www.youtube.com/watch?v=7hx4gdlfamo), we should try to include some logic in our break-up prediction---luckily we spent the last 16 weeks learning to do just that!
#
# We have provided all the code you need to complete the analyses here, which does the following:
# 1. Defines functions to help with the analysis
# 2. Imports data and prepares it for use with the functions
# 3. Illustrates how the functions can be used to fit a distribution and compute probabilities for break-up occurring in specific increments of time (assuming the Normal distribution)
#
# The time information can be used in three ways (illustrated below), each of which can be interpreted as random variables. The data for these three random variables are processed as follows for all break-up observations:
# 1. minutes from April 1, 00:00 (e.g., 2000 corresponds to April, 2 at 09:20)
# 2. minutes during any day (e.g., 875 corresponds to 14:35)
# 3. days from April 1 (e.g., 16 corresponds to April 16)
#
# The basic unit is decimal minutes (type float) from a specific `reference_date`. The documentation and code below also tries to follow two conventions:
# 1. The three time-related units use function and variable names `date` `time` and `day`, repsectively
# 2. In the functions, `minute` is the time (e.g., the MM part of a time HH:MM), wherease `minutes` (note the `s`) is a `float` representation of time in unit of decimal minutes
#
# Read through the code, to get an idea of what it is doing, then use the functions (and examples) provided to make the necessary computations.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 0:</b>  
#
# Read the documentation, code and examples provided. Functions are defined first, then examples.
# </p>
# </div>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar

# %% [markdown] id="0491cc69"
# A `reference_date` is used to find the relative time for each year, with midnight prior to the morning of April 1 set as default.

# %%
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

    # return print(f"The number of minutes passed in the day is {minutes}")
    # return print(f"{minutes:.1f} minutes corresponds to:",
    #              f"Month {month_name}, Day {new_date.day},",
    #              f"Hour {new_date.hour}, Minute {new_date.minute}.")
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

    # Calculate the CDF at minute1 and minute2
    cdf_start = norm.cdf(time_float_start, mu, std)
    cdf_end = norm.cdf(time_float_end, mu, std)

    return cdf_end - cdf_start


# %% [markdown]
# ## Import Data
#
# Data has already been cleaned and prepared for calculations; the following imports it into a DataFrame and sets two columns to datetime. In particular the columns `minutes` and `minutes_in_day` are the ones of insterest. `minutes` contains floats with the ammount of minutes between the date of each year and the 1st of April of that same year. `minutes_in_day` on the other hand provides the minutes in the day with respect to 00:00 of the `reference_date` given.

# %%
ice_data = pd.read_csv('data.csv')
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
ice_data

# %% [markdown]
# ## Time Unit Type 1: Minutes from the Reference Date
#
# The first two functions handle minutes data with respect to the `reference_date`.

# %%
date_to_minutes('4', '15', '10', '30')
date_to_minutes(4, 15, 10, 30)
date_to_minutes('4', '15', '10', '30', return_float=True)

# %%
minutes_to_date(2000)

# %% [markdown]
# It is always good to check that the functions work as expected. Here we check the `reference_date` and May 1.

# %%
date_to_minutes(4, 1, 0, 0)
minutes_to_date(date_to_minutes(4, 1, 0, 0, return_float=True))
minutes_to_date(date_to_minutes(5, 1, 0, 0, return_float=True))

# %% [markdown]
# This cell illustrates how you can define your `data` as the minutes (DataFrame column `minutes`) since `reference_date` and fit a Normal distribution.
#
# The empirical and fitted PDF's are also plotted.

# %%
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

# %% [markdown]
# Once the fit is done, the next step would be to calculate some probabilities using `prob_between_minutes`.

# %%
test_min_1 = 26670
test_min_2 = 26671
print(f'Function prob_between_floats returns type: ',
      f'{type(prob_between_floats(test_min_1, test_min_2, mu_min, std_min))}')
print(f'Probability between {test_min_1} min and {test_min_2} min',
      f'is {prob_between_floats(test_min_1, test_min_2, mu_min, std_min):.3e}')
prob_between_floats(test_min_1, test_min_2, mu_min, std_min)

# %% [markdown]
# ## Time Unit Type 2: Time During a Day
#
# Similar to what you saw in the previous section, the same procedure can be applied to the minutes in the day; variable names here use the word `time` to distinguish between minutes from `reference_date` and minutes during a day. The function `minutes_to_time` again provides you with a transformation from minutes to hours and minutes, and the function `time_to_minutes` does the opposite.

# %%
minutes_to_time(875)

# %%
time_to_minutes('14', '35')
time_to_minutes(14, 35)

# %% [markdown] id="0491cc69"
# A probability distribution is fit, as before.

# %%
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

# %% [markdown]
# For this case the same function prob_between_minutes can be used. Try making you own guesses!

# %%
prob_between_floats(875, 876, mu_time, std_time)

# %% [markdown] id="0491cc69"
# ## Time Unit Type 3: Days from the Reference Date
#
# This example is similar to the first case, except `days` from the `reference_date` are used rather than minutes.

# %%
days = (ice_data['datetime'] - ice_data['ref_date_annual']).dt.total_seconds()/(60*60*24)
data = days

# Calculate the Gaussian fitted PDF
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

# %% [markdown]
# Computing probability for a given day is easy, since the mean and standard deviation are found using days and the distribution is scaled automatically.

# %%
prob_between_floats(31, 32, mu_time_as_day, std_time_as_day)

# %% [markdown]
# ## Task 1: Compute a few warm-up probabilities
#
# Use the functions illustrated above to complete the following tasks. First, another example for how you can define your guess for day and time
#

# %%
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

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.1:</b>  
#
# Given a specific day, hour and minute, find the probability of being correct, assuming the Normal distribution.
# </p>
# </div>

# %%
p1 = prob_between_floats(prediction_min,
                         prediction_min+1,
                         mu_min, std_min)
print(f'Probability of my guess is: {p1:.3e}')

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.2:</b>  
#
# Given a specific hour and minute in any day, find the probability of being correct, assuming the Normal distribution.
# </p>
# </div>

# %%
p2 = prob_between_floats(prediction_min_in_day,
                         prediction_min_in_day+1,
                         mu_time, std_time)
print(f'Probability of my guess is: {p2:.3e}')

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.3:</b>  
#
# Given a specific day, find the probability of being correct, assuming the Normal distribution.
#     
# <em>Remember that no functions are provided for Type 3 time units, so if your guess is in May you will need to continue numbering from the 30th day in April (e.g., May 2 = day 32).</em>
# </p>
# </div>

# %%
p3 = prob_between_floats(my_day,
                         my_day+1,
                         mu_time_as_day,
                         std_time_as_day)
print(f'Probability of my guess is: {p3:.3e}')

# %% [markdown]
# ## Task 2: Compute the probability of your prediction!
#
# Computing the probability that your prediction is correct is not as straightforward as choosing one of the methods above. It is actually related to system probability!
#
# To find the probability of your prediction, you need to combine 2 of the 3 pieces used in Task 1.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 2:</b>  
#
# Compute the probability of your prediction being correct (you can assume the two "things" you need are independent).
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Solution:</b>   
#
# Type 1 is not good: it assumes time as a continuous across many days, however, from Type 2 it is clear that most break-up times happen in the early afternoon. To account for this we should separate the prediction probability into two parts: day and time of day, which is a combination of Type 2 and Type 3. The probability of our prediction being correct is the intersection of these two types of predictions being correct.
# </p>
# </div>

# %%
p_prediction = p2*p3
print(f'Probability of my prediction is: {p_prediction:.3e}')

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
