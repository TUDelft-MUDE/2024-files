# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GA 1.3: Warmup Notebook
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3. September 20, 2024.*
#
# This notebook is designed to help you prepare for GA 1.3, as we do not release the entire assignment in advance. This is formatted similarly to a PA and is meant for you to read, practice and explore - we hope you find it useful!

# %% [markdown]
# <div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p><b>Note:</b> this is the first time we have tried a "warmup" notebook in MUDE, so the format and scope of this activity may change throughout the semester.</p></div>

# %%
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# %% [markdown]
# ## Part 1: Dictionary Review
#
# Several functions in GA 1.3 require the use of a Python dictionary to make it easier to keep track of important data, variables and results for the various _models_ we will be constructing and validating.
#
# _It may be useful to revisit PA 1.1, where there was a brief infroduction to dictionaires. That PA contains all the dictionary info you need for GA 1.3. A [read-only copy is here](https://mude.citg.tudelft.nl/2024/files/Week_1_1/PA_1_1_Catch_Them_All.html) and [the source code (notebook) is here](https://gitlab.tudelft.nl/mude/2024-week-1-1)._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# $\textbf{Task 1.1}$ 
#     
# Read and run the cell below to make sure you remember how to use a dictionary.
#
# Modify the function to print some of the other key-value pairs of the dictionary.
#     
# </p>
# </div>

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

    # ADD MORE CODE HERE

    if 'new_key' in d:
        print('new_key exists and has value:', d['new_key'])
    return

function_that_uses_my_dictionary(my_dictionary)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# $\textbf{Task 1.2}$ 
#
# Test your knowledge by adding a new key <code>new_key</code> and then executing the function to print the value.
#     
# </p>
# </div>

# %%
YOUR_CODE_HERE
function_that_uses_my_dictionary(my_dictionary)

# %% [markdown]
# **Hint**
#
# Once you make the modifications, you should be able to reproduce the following output (depending on your value for `new_key`):
#
# ```
# value1
# Dictionary Example
# [1, 2, 3]
# [1 2 3]
# hello
# new_key exists and has value: new_value
# ```

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# $\textbf{Task 1.3}$ 
#     
# Run the cell below to print all of the keys and the value <code>type</code>. Then modify the code to print the values directly.
#
# </p>
# </div>
#

# %%
print("Keys and Values (type):")
for key, value in my_dictionary.items():
    print(f"{key:16s} -->    {type(value)}")

# %% [markdown]
# ## Part 2: Importing Functions from a `*.py`file
#
# Sometimes it is useful to put code in `*.py`files. It is very easy to import the contents of these files into a notebook. The code cell below imports the entire contents of file `warmup.py` into the notebook. 

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# $\textbf{Task 2}$ 
#     
# Open the file and read the contents. Then run the cell below to use the function that is inside it to confirm that it is imported properly.
#
# </p>
# </div>
#

# %%
from warmup import *

# %%
YOUR_CODE_HERE

# %% [markdown]
# ## Part 3: Data Import and Interpolation
#
# This Part illustrates the process of importing datasets, doing a bit of manipulation to both the _observations_ and the _times_, then doing a bit of interpolation. The code will run without error as-is, so first you should read each step and try to understand the functions/methods used. Then, modify the code to explore each  object; in particular, use the `print`, `type` and `shape` methods.

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

# DO MORE STUFF TO EXPLORE WHAT IS IN THE DATA!
      

# %% [markdown]
# You may have noticed that the two datasets are not made at identical time increments. This is a problem if we want to compare them, or use them together in an analysis where values of the observations are needed at identical observation times. You will therefore have to *interpolate* the data to the same times for a further analysis. You can use the SciPy function ```interpolate.interp1d``` (read its [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)).
#
# The cells below do the following:
# 1. Define a function to convert the time unit
# 2. Convert the time stamps for all data
# 3. Use `interp1d` to interpolate the measurements of `dataset2` at the time of the measurements of `dataset1`

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
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
