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
idx_max = P.groupby(pd.DatetimeIndex(P['Date']).year)['Prec'].idxmax()
YM = P.loc[idx_max]
print('The shape of the sampled extremes is:', YM.shape)
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(YM['Date'], YM['Prec'], 40, 'r')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
def ecdf(var):
    x = np.sort(var)
    n = x.size
    y = np.arange(1, n + 1)/(n + 1)
    return [y, x]
params_YM = stats.genextreme.fit(YM['Prec'])
print('GEV parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_YM))
x_range = np.linspace(0, 750, 100)
plt.figure(figsize=(10, 6))
plt.step(ecdf(YM['Prec'])[1],
         1 - ecdf(YM['Prec'])[0],
         'cornflowerblue',
         label = 'Yearly maxima')
plt.plot(x_range,
         1 - stats.genextreme.cdf(x_range, *params_YM),
         '--k', label='GEV')
if params_YM[0]>0:
    bound = params_YM[1] - params_YM[2]/(-params_YM[0])
    plt.axvline(x = bound, color = 'red',
                linestyle = ':',
                label='Bound')
plt.xlabel('Precipitation [mm]')
plt.ylabel('Exceedance probability, $P[X > x]$')
plt.yscale('log') 
plt.grid()
plt.legend()
plt.title('Empirical and YM/GEV Distributions, Precipitation')
plt.tight_layout()
print('GEV parameters are: {0:.3f} | {3:.3f} | {1:.3f} | {2:.3f}\n'.format(*params_YM, -params_YM[0]))
if params_YM[0]>0:
    bound = params_YM[1] - params_YM[2]/(-params_YM[0])
    print(f'Shape parameter from scipy.stats is {params_YM[0]:.3f}\n'
          '  - scipy.stats shape greater than 0\n'
          '  - MUDE book shape less than 0\n'
          '  - Tail type is Reverse Weibull --> there is a bound!\n'
          '  - bound = '
          f'{params_YM[1]:.3f} - {params_YM[2]:.3f}/(-{params_YM[0]:.3f}) = {bound:.3f}')
elif params_YM[0]<0:
    print(f'Shape parameter from scipy.stats is {params_YM[0]:.3f}\n'
          '  - scipy.stats shape less than 0\n'
          '  - MUDE book shape greater than 0\n'
          '  - Tail type is Frechet --> unbounded\n')
else:
     print(f'Shape parameter from scipy.stats is {params_YM[0]:.3f}\n'
          '  - Tail type is Gumbel\n')
threshold = 40
distance = 2 #days
peaks, _ = find_peaks(P['Prec'], height=threshold, distance=distance)
print('The shape of the sampled extremes is:', peaks.shape)
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(P.iloc[peaks, 0], P.iloc[peaks, 1], 40, 'cornflowerblue')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
params_POT = stats.genpareto.fit(P.iloc[peaks, 1] - threshold, floc=0)
print('GPD parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_POT))
excess_range = np.linspace(0, 750, 500)
plt.figure(figsize=(10, 6))
plt.step(ecdf(P.iloc[peaks, 1])[1],
         1 - ecdf(P.iloc[peaks, 1])[0],
         'cornflowerblue', label = 'POT')
plt.plot(excess_range + threshold,
         1 - stats.genpareto.cdf(excess_range, *params_POT),
         '--k', label='GPD')
if params_POT[0]<0:
    bound_POT = threshold - params_POT[2]/params_POT[0]
    print(f'Bound exists at {bound_POT:.3f}')
    plt.axvline(x = bound_POT, color = 'red',
                linestyle = ':',
                label='Bound')
plt.xlabel('Precipitation [mm]')
plt.ylabel('Exceedance probability, $P[X > x]$')
plt.yscale('log') 
plt.grid()
plt.legend()
plt.title('Empirical and POT/GPD Distributions, Precipitation')
plt.tight_layout()
print('GPD parameters are: {0:.3f} | {1:.3f} | {2:.3f}\n'.format(*params_POT, params_POT[0]))
if params_POT[0]>0:
    print(f'Shape parameter from scipy.stats is {params_POT[0]:.3f}\n'
          '  - scipy.stats and MUDE book shape greater than 0\n'
          '  - Tail type --> heavy; power function behavior\n')
elif params_POT[0]==-1:
    print(f'Shape parameter from scipy.stats is {params_POT[0]:.3f}\n'
          '  - Tail type is Exponential\n')
elif params_POT[0]<0:
    print(f'Shape parameter from scipy.stats is {params_POT[0]:.3f}\n'
          '  - scipy.stats and MUDE book shape less than 0\n'
          '  - Tail type --> bounded\n'
          '  - bound = '
          f'{threshold:.3f} - {params_POT[2]:.3f}/({params_POT[0]:.3f}) = {bound_POT:.3f}')
else:
     print(f'Shape parameter from scipy.stats is {params_POT[0]:.3f}\n'
          '  - Tail type is Gumbel\n')
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(P.iloc[peaks, 0], P.iloc[peaks, 1],
            50, 'cornflowerblue', label='POT')
plt.scatter(YM['Date'], YM['Prec'],
            60, 'r', marker='x', label='BM')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
plt.legend()
plt.title('Samples from BM and POT Methods, Precipitation')
plt.tight_layout()
return_period_range = np.linspace(1, 800, 500)
YM_range = stats.genextreme.ppf(1 - 1/return_period_range, *params_YM)
average_n_excesses = len(peaks)/YM.shape[0]
non_exc_prob_range = 1 - 1/(return_period_range*average_n_excesses)
POT_range = stats.genpareto.ppf(non_exc_prob_range, *params_POT) + threshold
plt.figure(figsize=(10, 6))
plt.plot(YM_range, return_period_range, '--r', label = 'YM&GEV', linewidth=5)
plt.step(ecdf(YM['Prec'])[1], 1/(1 - ecdf(YM['Prec'])[0]),
         ':r', label = ' ECDF Yearly maxima')
plt.plot(POT_range, return_period_range, 'cornflowerblue', label = 'POT&GPD', linewidth=5)
plt.step(ecdf(P.iloc[peaks, 1])[1], 1/((1 - ecdf(P.iloc[peaks, 1])[0])*average_n_excesses),
         linestyle=':', color = 'cornflowerblue', label = 'ECDF POT')
plt.plot([771, 771],[0, 1000], '-.k', label = 'Event of October 2024')
if params_YM[0]>0:
    bound_YM = params_YM[1] - params_YM[2]/(-params_YM[0])
    plt.axvline(x = bound_YM, color = 'black',
                linestyle = ':',
                label='Bound, YM',
                linewidth=3)
if params_POT[0]<0:
    bound_POT = threshold - params_POT[2]/params_POT[0]
    print(f'Bound exists at {bound_POT:.3f}')
    plt.axvline(x = bound_POT, color = 'black',
                linestyle = ':',
                label='Bound, POT',
                linewidth=3)
plt.xlabel('Precipitation [mm]')
plt.ylabel('RT [years]')
plt.yscale('log') 
plt.ylim([1, 1000])
plt.grid()
plt.legend()
plt.title('Return Period and Design Values, Rain')
plt.tight_layout()
YM_RT = 1/(1-stats.genextreme.cdf(771, *params_YM))
average_n_excesses = len(peaks)/YM.shape[0]
POT_prob = stats.genpareto.cdf(771 - threshold, *params_POT)
RT_POT = 1/((1-POT_prob)*average_n_excesses)
print(f'The return period obtained with YM+GEV is {YM_RT:.3f}\n'
          f'The return period obtained with POT+GPD is {RT_POT:.3f}\n')
