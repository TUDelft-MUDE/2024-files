
import numpy as np
from scipy import interpolate
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

from functions import *

np.set_printoptions(precision=3)

my_dictionary = {'key1': 'value1',
                 'key2': 'value2',
                 'name': 'Dictionary Example',
                 'a_list': [1, 2, 3],
                 'an_array': np.array([1, 2, 3]),
                 'a_string': 'hello'
                 }

def function_that_uses_my_dictionary(d):
    print(d['key1'])

    # SOLUTION:
    print(d['name'])
    print(d['a_list'])
    print(d['an_array'])
    print(d['a_string'])

    if 'new_key' in d:
        print('new_key exists and has value:', d['new_key'])
    return

function_that_uses_my_dictionary(my_dictionary)

YOUR_CODE_HERE
function_that_uses_my_dictionary(my_dictionary)

gnss = pd.read_csv('./data/gnss_observations.csv')
times_gnss = pd.to_datetime(gnss['times'])
y_gnss = (gnss['observations[m]']).to_numpy()*1000

insar = pd.read_csv('./data/insar_observations.csv')
times_insar = pd.to_datetime(insar['times'])
y_insar = (insar['observations[m]']).to_numpy()*1000

gw = pd.read_csv('./data/groundwater_levels.csv')
times_gw = pd.to_datetime(gw['times'])
y_gw = (gw['observations[mm]']).to_numpy()

YOUR_CODE_HERE

def to_days_years(times):
    '''Convert the observation times to days and years.'''
    
    times_datetime = pd.to_datetime(times)
    time_diff = (times_datetime - times_datetime[0])
    days_diff = (time_diff / np.timedelta64(1,'D')).astype(int)
    
    days = days_diff.to_numpy()
    years = days/365
    
    return days, years

days_gnss,  years_gnss  = to_days_years(times_gnss)
days_insar, years_insar = to_days_years(times_insar)
days_gw,    years_gw    = to_days_years(times_gw)

interp = interpolate.interp1d(days_gw, y_gw)

GW_at_GNSS_times = interp(days_gnss)
GW_at_InSAR_times = interp(days_insar)

YOUR_CODE_HERE

plt.figure(figsize=(15,5))
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         'o', mec='black', label = 'GNSS')
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         'o', mec='black', label = 'InSAR')
plt.legend()
plt.ylabel('Displacement [mm]')
plt.xlabel('Time')
plt.show()

model_insar = {'data_type': 'InSAR',
               'y':y_insar,
               'times':times_insar,
               'groundwater': GW_at_InSAR_times
               }

model_gnss = {'data_type': 'GNSS',
               'y':y_gnss,
               'times':times_gnss,
               'groundwater': GW_at_GNSS_times
               }

YOUR_CODE_HERE

model_insar['A'] = YOUR_CODE_HERE
model_gnss['A'] = YOUR_CODE_HERE

model_insar['Sigma_Y'] = YOUR_CODE_HERE
model_gnss['Sigma_Y'] = YOUR_CODE_HERE

def BLUE(d):
    """Calculate the Best Linear Unbiased Estimator
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    y = d['y']
    A = d['A']
    Sigma_Y = d['Sigma_Y']

    Sigma_X_hat = YOUR_CODE_HERE
    x_hat = YOUR_CODE_HERE
    
    y_hat = YOUR_CODE_HERE

    e_hat = YOUR_CODE_HERE

    Sigma_Y_hat = YOUR_CODE_HERE
    std_Y_hat = YOUR_CODE_HERE

    Sigma_e_hat = YOUR_CODE_HERE
    std_e_hat = YOUR_CODE_HERE

    d['Sigma_X_hat'] = Sigma_X_hat
    d['x_hat'] = x_hat
    d['y_hat'] = y_hat
    d['e_hat'] = e_hat
    d['Sigma_Y_hat'] = Sigma_Y_hat
    d['std_Y_hat'] = std_Y_hat
    d['Sigma_e_hat'] = Sigma_e_hat
    d['std_e_hat'] = std_e_hat

    return d

model_insar = BLUE(model_insar)
x_hat_insar = model_insar['x_hat']

YOUR_CODE_HERE

model_gnss = BLUE(model_gnss)
x_hat_gnss = model_gnss['x_hat']

YOUR_CODE_HERE

Sigma_X_hat_insar = model_insar['Sigma_X_hat']

YOUR_CODE_HERE

Sigma_X_hat_gnss = model_gnss['Sigma_X_hat']

YOUR_CODE_HERE

def get_CI(d, alpha):
    """Compute the confidence intervals.
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    std_e_hat = d['std_e_hat']
    std_Y_hat = d['std_Y_hat']

    k = YOUR_CODE_HERE
    CI_Y_hat = YOUR_CODE_HERE
    CI_res = YOUR_CODE_HERE

    d['alpha'] = alpha
    d['CI_Y_hat'] = CI_Y_hat
    d['CI_res'] = CI_res

    return d

model_insar = YOUR_CODE_HERE
model_gnss = YOUR_CODE_HERE

print("Keys and Values (type) for model_insar:")
for key, value in model_insar.items():
    print(f"{key:16s} -->    {type(value)}")
print("\nKeys and Values (type) for model_gnss:")
for key, value in model_gnss.items():
    print(f"{key:16s} -->    {type(value)}")

_, _ = plot_model(YOUR_CODE_HERE)

_, _ = plot_residual(YOUR_CODE_HERE)

_, _ = plot_residual_histogram(YOUR_CODE_HERE)

