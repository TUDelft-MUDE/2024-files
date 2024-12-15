# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# %% [markdown]

# %% [markdown]

# %%
my_dictionary = {'key1': 'value1',
                 'key2': 'value2',
                 'name': 'Dictionary Example',
                 'a_list': [1, 2, 3],
                 'an_array': np.array([1, 2, 3]),
                 'a_string': 'hello'
                 }

def function_that_uses_my_dictionary(d):
    print(d['key1'])

    

    if 'new_key' in d:
        print('new_key exists and has value:', d['new_key'])
    return

function_that_uses_my_dictionary(my_dictionary)

# %% [markdown]

# %%
YOUR_CODE_HERE
function_that_uses_my_dictionary(my_dictionary)

# %% [markdown]

# %% [markdown]

# %%
print("Keys and Values (type):")
for key, value in my_dictionary.items():
    print(f"{key:16s} -->    {type(value)}")

# %% [markdown]

# %% [markdown]

# %%
from warmup import *

# %%
YOUR_CODE_HERE

# %% [markdown]

# %%
dataset1 = pd.read_csv('./data_warmup/dataset1.csv')
times1 = pd.to_datetime(dataset1['times'])
obs1 = (dataset1['observations[m]']).to_numpy()*1000

dataset2 = pd.read_csv('./data_warmup/dataset2.csv')
times2 = pd.to_datetime(dataset2['times'])
obs2 = (dataset2['observations[mm]']).to_numpy()

# %%
print(type(dataset1), '\n',
      type(dataset2), '\n',
      type(times1), '\n',
      type(times2), '\n',
      type(obs1), '\n',
      type(obs2))

print(np.shape(dataset1), '\n',
      np.shape(dataset2), '\n',
      np.shape(times1), '\n',
      np.shape(times2), '\n',
      np.shape(obs1), '\n',
      np.shape(obs2))

      

# %% [markdown]

# %%
def to_days_years(times):
    '''Convert the observation times to days and years.'''
    
    times_datetime = pd.to_datetime(times)
    time_diff = (times_datetime - times_datetime[0])
    days_diff = (time_diff / np.timedelta64(1,'D')).astype(int)
    
    days = days_diff.to_numpy()
    years = days/365
    
    return days, years

# %%
days1,  years1  = to_days_years(times1)
days2,  years2  = to_days_years(times2)

interp = interpolate.interp1d(days2, obs2)

obs2_at_times_for_dataset1 = interp(days1)

print(type(obs2_at_times_for_dataset1), '\n',
      np.shape(obs2_at_times_for_dataset1), '\n',
      obs2_at_times_for_dataset1)

# %% [markdown]

