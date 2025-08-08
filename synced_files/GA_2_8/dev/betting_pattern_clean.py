import pandas as pd
import numpy as np
from datetime import time ,datetime,timedelta
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # stuff to ptu the tick in the ylabel in thousand of dollars
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
bets_2017=pd.read_csv('parsed_df_v2.csv',index_col=0)
bets_2018=pd.read_csv('parsed_df.csv',index_col=0)
breakup_date_2018 = pd.to_datetime('2018-05-01 13:18')
breakup_day_2018 = breakup_date_2018.dayofyear
breakup_hour_2018 = breakup_date_2018.hour
breakup_hour_decimal_2018 = breakup_date_2018.hour + breakup_date_2018.minute / 60
break_up_date_2017 = pd.to_datetime('5/1/2017  12:00:00 PM')
breakup_day_2017 = break_up_date_2017.dayofyear
breakup_hour_2017 = break_up_date_2017.hour
breakup_hour_decimal_2017 = break_up_date_2017.hour + break_up_date_2017.minute / 60
bets_2017.index=pd.to_datetime(bets_2017.index)
bets_2018.index=pd.to_datetime(bets_2018.index)
bets_2017['DayOfYear'] = bets_2017.index.dayofyear
bets_2018['DayOfYear'] = bets_2018.index.dayofyear
bets_2017['Hour'] = bets_2017.index.hour
bets_2018['Hour'] = bets_2018.index.hour
bets_2017['minute'] = bets_2017.index.minute
bets_2018['minute'] = bets_2018.index.minute
bets_2017['Hour_decimal'] = bets_2017.index.hour + bets_2017.index.minute / 60
bets_2018['Hour_decimal'] = bets_2018.index.hour + bets_2018.index.minute / 60
bets_2017['Same_Day'] = bets_2017.groupby('DayOfYear')['DayOfYear'].transform('size')
bets_2018['Same_Day'] = bets_2018.groupby('DayOfYear')['DayOfYear'].transform('size')
bets_2017['Same_decimal_hour'] = bets_2017.groupby('Hour_decimal')['Hour_decimal'].transform('size')
bets_2018['Same_decimal_hour'] = bets_2018.groupby('Hour_decimal')['Hour_decimal'].transform('size')    
plt.figure(figsize=(20, 6))
plt.scatter(bets_2017['DayOfYear'], bets_2017['Same_Day'], color='blue', alpha=0.7,label='2017')
plt.scatter(bets_2018['DayOfYear'], bets_2018['Same_Day'], color='cyan', alpha=0.7,label='2018')
plt.title('Count of Bets by Day of Year', fontsize=16)
plt.xlabel('Day of Year', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(breakup_day_2017,label='2017',color='green',linestyle='--')
plt.axvline(breakup_day_2018,label='2018',color='red',linestyle='--')
plt.xlim([80,150])
plt.legend()
plt.show()
plt.figure(figsize=(20, 6))
plt.scatter(bets_2017['Hour_decimal'], bets_2017['Same_decimal_hour'], color='red', alpha=0.7, label='2017')
plt.scatter(bets_2018['Hour_decimal'], bets_2018['Same_decimal_hour'], color='orange', alpha=0.7, label='2018')
plt.title('Count of Bets by Hour/minute of Day', fontsize=16)
plt.xlabel('Deciaml hour of Day', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(breakup_hour_decimal_2017,color='red',linestyle='--')
plt.axvline(breakup_hour_decimal_2018,color='orange',linestyle='--')
plt.legend()
plt.xlim([0,24])
plt.xticks(np.arange(0, 24, 1))
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
heatmap_data_4, xedges_4, yedges_4 = np.histogram2d(bets_2017['DayOfYear'], bets_2017['Hour_decimal'], bins=(np.arange(1, 366), np.arange(0, 24, 0.1)))
im1 = ax1.imshow(heatmap_data_4.T, aspect='auto', cmap='viridis', origin='lower', extent=[xedges_4[0], xedges_4[-1], yedges_4[0], yedges_4[-1]])
ax1.set_title('2017: Distribution of Bets by Day of Year and Hour/Minute of Day', fontsize=16)
ax1.set_xlabel('Day of Year', fontsize=14)
ax1.set_ylabel('Decimal Hour of Day', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim([100, 140])
ax1.set_ylim([8, 22])
heatmap_data_4, xedges_4, yedges_4 = np.histogram2d(bets_2018['DayOfYear'], bets_2018['Hour_decimal'], bins=(np.arange(1, 366), np.arange(0, 24, 0.1)))
im2 = ax2.imshow(heatmap_data_4.T, aspect='auto', cmap='viridis', origin='lower', extent=[xedges_4[0], xedges_4[-1], yedges_4[0], yedges_4[-1]])
ax2.set_title('2018: Distribution of Bets by Day of Year and Hour/Minute of Day', fontsize=16)
ax2.set_xlabel('Day of Year', fontsize=14)
ax2.set_ylabel('Decimal Hour of Day', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlim([100, 140])
ax2.set_ylim([8, 22])
fig.colorbar(im1, ax=ax1, label='Number of Bets')
fig.colorbar(im2, ax=ax2, label='Number of Bets')
plt.tight_layout()
plt.show()
bets_2017_count = bets_2017['Name'].value_counts()
n_unique_2017 = len(bets_2017['Name'].unique())
print(f'2017: Number of unique persons: {n_unique_2017}')
print(bets_2017_count.head(10))
bets_2018_count = bets_2018['Name'].value_counts()
n_unique_2018 = len(bets_2018['Name'].unique())
print(f'2018: Number of unique persons: {n_unique_2018}')
print(bets_2018_count.head(10))
plt.figure(figsize=(20, 6))
plt.hist(bets_2017_count.values, bins=range(0, bets_2017_count.max() + 2), 
         color='blue', alpha=0.6, density=True, label='2017')
plt.hist(bets_2018_count.values, bins=range(0, bets_2018_count.max() + 2), 
         color='cyan', alpha=0.6, density=True, label='2018')
plt.xlabel('Number of Bets', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Density Distribution of Number of Bets per Person', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim([0, 20])
plt.xticks([0, 5, 10, 15, 20])
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 10))
plt.title('Betting pattern of `Whales`')
plt.scatter(bets_2017[bets_2017['Name']=='JOSEPH DINKINS']['DayOfYear'],bets_2017[bets_2017['Name']=='JOSEPH DINKINS']['Hour_decimal'],label='JOSEPH DINKINS',color='blue',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='THE HOOPER FAMILY POOL']['DayOfYear'],bets_2017[bets_2017['Name']=='THE HOOPER FAMILY POOL']['Hour_decimal'],label='THE HOOPER FAMILY POOL',color='red',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='DESMOND DUFFY']['DayOfYear'],bets_2017[bets_2017['Name']=='DESMOND DUFFY']['Hour_decimal'],color='cyan',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='IVORY JACKS']['DayOfYear'],bets_2017[bets_2017['Name']=='IVORY JACKS']['Hour_decimal'],color='green',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='VINCENT JONES']['DayOfYear'],bets_2017[bets_2017['Name']=='VINCENT JONES']['Hour_decimal'],color='purple',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='SCOTT HALAMA']['DayOfYear'],bets_2017[bets_2017['Name']=='SCOTT HALAMA']['Hour_decimal'],color='gold',alpha=0.6)
plt.scatter(bets_2017[bets_2017['Name']=='SANDRA BJELLAND']['DayOfYear'],bets_2017[bets_2017['Name']=='SANDRA BJELLAND']['Hour_decimal'],color='green',alpha=0.6)
plt.xlabel('Day of Year')
plt.ylabel('Decimal Hour of Day')
print(f"Desmond Duffy : {bets_2017[bets_2017['Name']=='O'].index }")
N=10
top_N = bets_2017_count[0:N].index
colors = plt.cm.tab20.colors
plt.figure(figsize=(20, 10))
plt.title('2017: Betting Pattern of `Whales`')
for i, name in enumerate(top_N):
    color = colors[i % len(colors)]  #e
    plt.scatter(
        bets_2017[bets_2017['Name'] == name]['DayOfYear'],
        bets_2017[bets_2017['Name'] == name]['Hour_decimal'],
        label=name,
        color=color,
        alpha=0.6
    )
plt.axhline(breakup_hour_decimal_2017,label='2017',color='magenta',linestyle='--',linewidth=3)
plt.axvline(breakup_day_2017,label='2017',color='magenta',linestyle='--',linewidth=3)
plt.xlabel('Day of Year')
plt.ylabel('Hour (Decimal)')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.grid()
plt.show()
N=10
top_N = bets_2018_count[0:N].index
colors = plt.cm.tab20.colors
plt.figure(figsize=(20, 10))
plt.title('2018: Betting Pattern of `Whales`')
for i, name in enumerate(top_N):
    color = colors[i % len(colors)]  #e
    plt.scatter(
        bets_2018[bets_2018['Name'] == name]['DayOfYear'],
        bets_2018[bets_2018['Name'] == name]['Hour_decimal'],
        label=name,
        color=color,
        alpha=0.6
    )
plt.axvline(breakup_day_2018,label='2018',color='magenta',linestyle='--',linewidth=3)
plt.axhline(breakup_hour_decimal_2018,label='2018',color='magenta',linestyle='--',linewidth=3)
plt.xlabel('Day of Year')
plt.ylabel('Hour (Decimal)')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
