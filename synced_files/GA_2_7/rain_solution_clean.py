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
            40, 'cornflowerblue', label='POT')
plt.scatter(YM['Date'], YM['Prec'],
            40, 'r', label='BM')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
plt.legend()
plt.title('Samples from BM and POT Methods, Precipitation')
plt.tight_layout()
YM_design_value = stats.genextreme.ppf(1 - 1/100, *params_YM)
average_n_excesses = len(peaks)/YM.shape[0]
non_exc_prob = 1 - 1/(100*average_n_excesses)
POT_design_value = stats.genpareto.ppf(non_exc_prob, *params_POT) + threshold
print('The design value for a RT = 100 years computed using',
      'BM and GEV is:', np.round(YM_design_value, 3), 'mm')
print('The design value for a RT = 100 years computed using',
      'POT and GPD is:', np.round(POT_design_value, 3), 'mm')
RT_range = np.linspace(1, 500, 500)
RT_range = np.linspace(2, 500, 500)
YM_range = stats.genextreme.ppf(1 - 1/RT_range, *params_YM)
average_n_excesses = len(peaks)/YM.shape[0]
non_exc_prob_range = 1 - 1/(RT_range*average_n_excesses)
POT_range = stats.genpareto.ppf(non_exc_prob_range, *params_POT) + threshold
plt.figure(figsize=(10, 6))
plt.plot(YM_range, RT_range, '--r', label = 'YM&GEV', linewidth=5)
plt.plot(POT_range, RT_range, 'cornflowerblue', label = 'POT&GPD', linewidth=5)
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
plt.grid()
plt.legend()
plt.title('Return Period and Design Values, Rain')
plt.tight_layout()
peaks_year = P.loc[peaks]
count = peaks_year.groupby(pd.DatetimeIndex(peaks_year['Date']).year)['Prec'].count()
mean_count = count.mean()
var_count = count.var()
print(f'Mean = {mean_count:.3f} and variance = {var_count:.3f}')
import math
k = np.arange(count.min(), count.max()+1,1)
PMF_calc = stats.poisson.pmf(k, mean_count)
occur = count.groupby(count).size()
occur = pd.DataFrame(occur, columns=['Prec'])
occur_df = pd.DataFrame(index=k, columns=['Count'])
occur_df = occur_df.join(occur)
occur_df = occur_df['Prec'].fillna(0)
PMF_data = occur_df.values/len(peaks_year)
chi = np.sum((PMF_data - PMF_calc)**2/PMF_calc)
p_value = 1 - stats.chi2.cdf(x=chi, df=1)
if p_value > 0.05:
    print('Accept the null hypothesis:\n',
          '  Poisson distribution can represent the excesses!')
else:
    print('Reject the null hypothesis:\n',
          '  Poisson distribution does NOT represent the excesses!')
plt.figure(figsize=(10, 6))
plt.hist(count, bins=25, label='Data', density=False)
plt.scatter(k, PMF_calc*len(peaks_year), label='Fit', color = 'k')
plt.legend()
plt.grid()
plt.title(f'Rain Case: p-value = {p_value:.3f}')
plt.ylabel('Frequency, observed excesses per year')
plt.xlabel('Number of times the precipitation '
           + f'{threshold} mm is exceeded in a year')
plt.tight_layout()
