import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
from math import ceil, trunc
plt.rcParams.update({'font.size': 14})
T = pd.read_csv('temp.csv', delimiter = ',', parse_dates = True).dropna().reset_index(drop=True)
T.columns=['Date', 'T'] #rename columns
T.head()
T['Date'] = pd.to_datetime(T['Date'], format='mixed')
T['Date']
fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
axes[0].hist(T['T'], label = 'T', density = True, edgecolor = 'darkblue')
axes[0].set_xlabel('PDF')
axes[0].set_xlabel('Maximum daily temperature [degrees]')
axes[0].grid()
axes[0].legend()
axes[0].set_title('(a) Histogram')
axes[1].plot(T['Date'], T['T'],'k', label = 'P')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Maximum daily temperature [degrees]')
axes[1].grid()
axes[1].set_title('(b) Time series')
axes[1].legend()
T['Year'] = T['Date'].dt.year
T['Month'] = T['Date'].dt.month
idx_max = #your code here
max_list = T.loc[idx_max]
def ecdf(var):
    return [y, x]
params_T = #your code here
print(params_T)
RT_range = #range of values of return level
monthly_probs = #compute monthly probabilities
eval_nitrogen = #compute the values of the random variable for those probabilities
plt.figure(figsize=(10, 6))
plt.plot(eval_nitrogen, RT_range, 'k')
plt.xlabel('Temperature [mm]')
plt.ylabel('RT [years]')
plt.yscale('log') 
plt.grid()
RT_design = #value of the return period
monthly_design = #compute the monthly probability
design_T = #compute the design value
print('The design value of temperature is:',
      f'{design_T:.3f}')
