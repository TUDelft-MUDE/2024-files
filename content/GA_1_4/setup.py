import os
import pickle
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

np.set_printoptions(precision=3)

# Initialize models and working directory

m1 = {'data_type': 'InSAR',
      'model_type': 'BLUE'}

m2 = {'data_type': 'GNSS',
      'model_type': 'BLUE'}

def get_path_aux(filename):
    directory = os.path.join(os.path.dirname(__file__), 'auxiliary_files')
    filepath = os.path.join(directory, filename)
    return os.path.normpath(filepath)


# Load and process data

insar = pd.read_csv(get_path_aux('insar_observations.csv'))
gnss = pd.read_csv(get_path_aux('gnss_observations.csv'))
gw = pd.read_csv(get_path_aux('groundwater_levels.csv'))

m1['times'] = pd.to_datetime(insar['times'])
m1['y'] = (insar['observations[m]']).to_numpy()*1000
m1['days'], years_insar = to_days_years(m1['times'])

m2['times'] = pd.to_datetime(gnss['times'])
m2['y'] = (gnss['observations[m]']).to_numpy()*1000
m2['days'], years_gnss  = to_days_years(m2['times'])

groundwater_data = {}
groundwater_data['times'] = pd.to_datetime(gw['times'])
groundwater_data['y'] = (gw['observations[mm]']).to_numpy()
groundwater_data['days'], _  = to_days_years(groundwater_data['times'])


interp = interpolate.interp1d(groundwater_data['days'],
                              groundwater_data['y'])
m1['groundwater'] = interp(m1['days'])
m2['groundwater'] = interp(m2['days'])

m1['groundwater_data'] = groundwater_data
m2['groundwater_data'] = groundwater_data

# INSAR

m1['A'] = np.ones((len(m1['times']), 3))
m1['A'][:,1] = m1['days']
m1['A'][:,2] = m1['groundwater']
m1['std_Y'] = 2 #mm
m1['Sigma_Y'] = np.identity(len(m1['times']))*m1['std_Y']**2
m1 = BLUE(m1)
m1 = get_CI(m1, 0.04)


# GNSS

m2['A'] = np.ones((len(m2['times']), 3))
m2['A'][:,1] = m2['days']
m2['A'][:,2] = m2['groundwater']
m2['std_Y'] = 15 #mm
m2['Sigma_Y'] = np.identity(len(m2['times']))*m2['std_Y']**2
m2 = BLUE(m2)
m2 = get_CI(m2, 0.04)

# Save models

m1_blue = m1
with open(get_path_aux('m1_blue.pickle'), 'wb') as file:
    pickle.dump(m1_blue, file)
print('\nModel 1 saved as m1_blue.pickle\n')
model_summary(m1_blue)

m2_blue = m2
with open(get_path_aux('m2_blue.pickle'), 'wb') as file:
    pickle.dump(m2_blue, file)
print('\nModel 2 saved as m2_blue.pickle\n')
model_summary(m2_blue)
