import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 14})
P = pd.read_csv('turis.csv', delimiter = ',', parse_dates = True)
P.columns=['Date', 'Prec'] #rename columns
P['Date'] = pd.to_datetime(P['Date'], format='mixed')
P = P.sort_values(by='Date')
P=P.reset_index(drop=True)
P.head()
print(f"{P['Prec'].size:d}",
      f"{P['Prec'].min():.3f}",
      f"{P['Prec'].max():.3f}",
      f"{P['Prec'].mean():.3f}",
      f"{P['Prec'].std():.3f}",
      f"{P['Prec'].isna().sum():d}",
      f"{sum(P['Prec']==0):d}",
      sep=' | ')
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
plt.title('Time series of precipitation');
idx_max = #your code here
YM = P.loc[idx_max]
print('The shape of the sampled extremes is:', YM.shape)
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(YM['Date'], YM['Prec'], 40, 'r')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
def ecdf(var):
    return [y, x]
params_YM = #your code here
print('GEV parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_YM))
threshold = 40
distance = 2 #days
peaks, _ = #your code here
print('The shape of the sampled extremes is:', peaks.shape)
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(P.iloc[peaks, 0], P.iloc[peaks, 1], 40, 'cornflowerblue')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
params_POT = #your code here
print('GPD parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_POT))
YM_RT = #return period obtained with YM
average_n_excesses = #average numbr of excesses per year
POT_prob = #non exceedance probability
RT_POT = #return period obtained with POT
print(f'The return period obtained with YM+GEV is {YM_RT:.3f}\n'
          f'The return period obtained with POT+GPD is {RT_POT:.3f}\n')
