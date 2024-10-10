## IMPORTS ##
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os

# This script is an excerpt from GA 1.4.
# There are two errors in this script. One is a logical error
# and the other is not. You will need to look at the figures 
# to identify the logical error, which is associated with the
# central difference calculation. The assert statement at the
# end will also check the values are correct. You may have to
# close the figure before the script runs until the end.

## FILE PATHS ##
# The first line is to make sure Python knows how to
# find files relative to the location of your script
file_path = os.path.join(os.path.dirname(__file__),
                         'justIce.csv')
data = pd.read_csv(filepath_or_buffer=file_path, index_col=0)
data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

## DATA ANALYSIS ##
data_2021 = data.loc['2021']
h_ice = (data_2021.to_numpy()).ravel()
t_days = ((data_2021.index - data_2021.index[0]).days).to_numpy()
dh_dt_FD = (h_ice[1:]-h_ice[:-1])/(t_days[1:]-t_days[:-1]) 
dh_dt_BD = (h_ice[1:]-h_ice[:-1])/(t_days[1:]-t_days[:-1]) 
dh_dt_CD = [(h_ice[i+1] - h_ice[i-1]) / (t_days[i+1] - t_days[i-1]) for i in range(0, len(t_days)-1)]

## PLOTTING ##
fig, ax1 = plt.subplots(figsize=(15,4))
ax1.scatter(t_days[:-1], dh_dt_FD,
            color='blue', marker='o', label='dh_dt_FE Forward Difference')
ax1.scatter(t_days[1:], dh_dt_BD,
            color='red', marker='o', label='dh_dt_BE Backward Difference')
ax1.scatter((t_days[1:]+t_days[:-1])/2, dh_dt_CD,
            color='purple', marker='o', label='dh_dt_CD Central Difference')
ax1.set_xlabel('Days')
ax1.set_ylabel('Growth Rate [cm/day]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()

ax2 = ax1.twinx()
ax2.scatter(t_days, h_ice, color='green', marker='x', label='h_ice Measurements')
ax2.set_ylabel('Ice thickness [cm]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc='upper left')

plt.show()

assert np.allclose(dh_dt_CD, [0.618, 0.346, 0.423,
                              0.222, -0.169, -0.508,
                              -0.127, -2.286, 2.540,
                              -0.063, -0.169, 0.127,
                              -0.169, -1.079],
                              rtol=1e-2), (
    "The central difference calculation is incorrect")
