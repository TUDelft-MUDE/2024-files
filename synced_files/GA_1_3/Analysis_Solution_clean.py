
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

my_dictionary['new_key'] = 'new_value'
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

data_list = ['y_gnss', 'y_insar', 'y_gw']
data_dict = {data_list[0]: y_gnss,
             data_list[1]: y_insar,
             data_list[2]: y_gw}
def print_summary(data):
    '''Summarize an array with simple print statements.'''
    print('Minimum =     ', data.min())
    print('Maximum =     ', data.max())
    print('Mean =        ', data.mean())
    print('Std dev =     ', data.std())
    print('Shape =       ', data.shape)
    print('First value = ', data[0])
    print('Last value =  ', data[-1])
    print('\n')
          
for item in data_list:
    print('Summary for array: ', item)
    print('------------------------------------------------')
    print_summary(data_dict[item])

times_dict = {data_list[0]: times_gnss,
              data_list[1]: times_insar,
              data_list[2]: times_gw}
def plot_data(times, data, label):
    plt.figure(figsize=(15,4))
    plt.plot(times, data, 'co', mec='black')
    plt.title(label)
    plt.xlabel('Times')
    plt.ylabel('Data [mm]')
    plt.show()

plt.figure(figsize=(15,4))
for i in range(3):
    plot_data(times_dict[data_list[i]],
              data_dict[data_list[i]],
              data_list[i])

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

print('array size of GW_at_GNSS_times', len(GW_at_GNSS_times))
print('array size of GW_at_InSAR_times', len(GW_at_InSAR_times))
print('array size of GW before interpolation', len(y_gw))

print('\nFirst values of times_gw:')
print(times_gw[0:2])
print('\nFirst values of y_gw:')
print(y_gw[0:2])
print('\nFirst values of times_gnss:')
print(times_gnss[0:2])
print('\nFirst values of GW_at_GNSS_times:')
print(GW_at_GNSS_times[0:2])

plt.figure(figsize=(15,5))
plt.plot(times_gnss, y_gnss, 'o', mec='black', label = 'GNSS')
plt.plot(times_insar, y_insar, 'o', mec='black', label = 'InSAR')
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

A_insar = np.ones((len(times_insar), 3))
A_insar[:,1] = days_insar
A_insar[:,2] = GW_at_InSAR_times

print ('The first 5 rows of the A matrix (InSAR) are:')
print (A_insar[0:5, :])

print ('The first 5 observations [mm] of y_insar are:')
print (y_insar[0:5])

m_insar = np.shape(A_insar)[0]
n_insar = np.shape(A_insar)[1]
print(f'm = {m_insar} and n = {n_insar}')

A_gnss = np.ones((len(times_gnss), 3))
A_gnss[:,1] = days_gnss
A_gnss[:,2] = GW_at_GNSS_times

print ('The first 5 rows of the A matrix (GNSS) are:')
print (A_gnss[0:5, :])

print ('\nThe first 5 observations [mm] of y_gnss are:')
print (y_gnss[0:5])

m_gnss = np.shape(A_gnss)[0]
n_gnss = np.shape(A_gnss)[1]
print(f'm = {m_gnss} and n = {n_gnss}')

model_insar['A'] = A_insar
model_gnss['A'] = A_gnss

print("Keys and Values (type) for model_insar:")
for key, value in model_insar.items():
    print(f"{key:16s} -->    {type(value)}")
print("\nKeys and Values (type) for model_gnss:")
for key, value in model_gnss.items():
    print(f"{key:16s} -->    {type(value)}")

std_insar = 2 #mm

Sigma_Y_insar = np.identity(len(times_insar))*std_insar**2

print ('Sigma_Y (InSAR) is defined as:')
print (Sigma_Y_insar)

std_gnss = 15 #mm (corrected from original value of 5 mm)

Sigma_Y_gnss = np.identity(len(times_gnss))*std_gnss**2

print ('\nSigma_Y (GNSS) is defined as:')
print (Sigma_Y_gnss)

model_insar['Sigma_Y'] = Sigma_Y_insar
model_gnss['Sigma_Y'] = Sigma_Y_gnss

def BLUE(d):
    """Calculate the Best Linear Unbiased Estimator
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    y = d['y']
    A = d['A']
    Sigma_Y = d['Sigma_Y']

    # Sigma_X_hat = YOUR_CODE_HERE
    # x_hat = YOUR_CODE_HERE
    
    # y_hat = YOUR_CODE_HERE

    # e_hat = YOUR_CODE_HERE

    # Sigma_Y_hat = YOUR_CODE_HERE
    # std_y = YOUR_CODE_HERE

    # Sigma_e_hat = YOUR_CODE_HERE
    # std_e_hat = YOUR_CODE_HERE

    # SOLUTION:
    Sigma_X_hat = np.linalg.inv(A.T @ np.linalg.inv(Sigma_Y) @ A)
    x_hat = Sigma_X_hat @ A.T @ np.linalg.inv(Sigma_Y) @ y
    
    y_hat = A @ x_hat

    e_hat = y - y_hat

    Sigma_Y_hat = A @ Sigma_X_hat @ A.T
    std_y = np.sqrt(Sigma_Y_hat.diagonal())

    Sigma_e_hat = Sigma_Y - Sigma_Y_hat
    std_e_hat = np.sqrt(Sigma_e_hat.diagonal())

    d['Sigma_X_hat'] = Sigma_X_hat
    d['x_hat'] = x_hat
    d['y_hat'] = y_hat
    d['e_hat'] = e_hat
    d['Sigma_Y_hat'] = Sigma_Y_hat
    d['std_y'] = std_y
    d['Sigma_e_hat'] = Sigma_e_hat
    d['std_e_hat'] = std_e_hat

    return d

model_insar = BLUE(model_insar)
x_hat_insar = model_insar['x_hat']

print ('The InSAR-estimated offset is', np.round(x_hat_insar[0],3), 'mm')
print ('The InSAR-estimated velocity is', np.round(x_hat_insar[1],4), 'mm/day')
print ('The InSAR-estimated velocity is', np.round(x_hat_insar[1]*365,4), 'mm/year')
print ('The InSAR-estimated GW factor is', np.round(x_hat_insar[2],3), '[-]\n')

model_gnss = BLUE(model_gnss)
x_hat_gnss = model_gnss['x_hat']

print ('The GNSS-estimated offset is', np.round(x_hat_gnss[0],3), 'mm')
print ('The GNSS-estimated velocity is', np.round(x_hat_gnss[1],4), 'mm/day')
print ('The GNSS-estimated velocity is', np.round(x_hat_gnss[1]*365,4), 'mm/year')
print ('The GNSS-estimated GW factor is', np.round(x_hat_gnss[2],3), '[-]')

Sigma_X_hat_insar = model_insar['Sigma_X_hat']

print ('Covariance matrix of estimated parameters (InSAR):')
print (Sigma_X_hat_insar)
print ('\nThe standard deviation for the InSAR-estimated offset is', 
       np.round(np.sqrt(Sigma_X_hat_insar[0,0]),3), 'mm')
print ('The standard deviation for the InSAR-estimated velocity is', 
       np.round(np.sqrt(Sigma_X_hat_insar[1,1]),4), 'mm/day')
print ('The standard deviation for the InSAR-estimated GW factor is', 
       np.round(np.sqrt(Sigma_X_hat_insar[2,2]),3), '[-]\n')

Sigma_X_hat_gnss = model_gnss['Sigma_X_hat']

print ('Covariance matrix of estimated parameters (GNSS):')
print (Sigma_X_hat_gnss)
print ('\nThe standard deviation for the GNSS-estimated offset is', 
       np.round(np.sqrt(Sigma_X_hat_gnss[0,0]),3), 'mm')
print ('The standard deviation for the GNSS-estimated velocity is', 
       np.round(np.sqrt(Sigma_X_hat_gnss[1,1]),4), 'mm/day')
print ('The standard deviation for the GNSS-estimated GW factor is', 
       np.round(np.sqrt(Sigma_X_hat_gnss[2,2]),3), '[-]')

def get_CI(d, alpha):
    """Compute the confidence intervals.
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    std_e_hat = d['std_e_hat']
    std_y = d['std_y']

    # k = YOUR_CODE_HERE
    # CI_y = YOUR_CODE_HERE
    # CI_res = YOUR_CODE_HERE

    # SOLUTION:
    k = norm.ppf(1 - 0.5*alpha)
    CI_y = k*std_y
    CI_res = k*std_e_hat
    CI_y_hat = k*np.sqrt(d['Sigma_Y_hat'].diagonal())

    d['alpha'] = alpha
    d['CI_y'] = CI_y
    d['CI_res'] = CI_res
    d['CI_Y_hat'] = CI_y_hat

    return d

model_insar = get_CI(model_insar, 0.04)
model_gnss = get_CI(model_gnss, 0.04)

print("Keys and Values (type) for model_insar:")
for key, value in model_insar.items():
    print(f"{key:16s} -->    {type(value)}")
print("\nKeys and Values (type) for model_gnss:")
for key, value in model_gnss.items():
    print(f"{key:16s} -->    {type(value)}")

_, _ = plot_model(model_insar)
_, _ = plot_model(model_gnss)

_, _ = plot_residual(model_insar)
_, _ = plot_residual(model_gnss)

_, _ = plot_residual_histogram(model_insar)
_, _ = plot_residual_histogram(model_gnss)

k_true = 0.15
R_true = -22 
a_true = 180
d0_true = 10

disp_insar = (d0_true + R_true*(1 - np.exp(-days_insar/a_true)) +
              k_true*GW_at_InSAR_times)
disp_gnss  = (d0_true + R_true*(1 - np.exp(-days_gnss/a_true)) +
              k_true*GW_at_GNSS_times)

plot_model(model_insar, alt_model=('True model', times_insar, disp_insar));
plot_model(model_gnss, alt_model=('True model', times_gnss, disp_gnss));

import ipywidgets as widgets
from ipywidgets import interact

def update_plot(x0, x1, x2):
    plt.figure(figsize=(15,5))
    for m in [model_gnss]: #[model_insar, model_gnss]:
        plt.plot(m['times'], m['y'], 'o', label=m['data_type'])
    plt.ylabel('Displacement [mm]')
    plt.xlabel('Time')
    
    y_fit = model_gnss['A'] @ [x0, x1, x2]
    if (x0 == 0) & (x1 == 0) & (x2 == 1):
        plt.plot(model_gnss['times'], y_fit, 'r', label='Groundwater data', linewidth=2)
    else:
        plt.plot(model_gnss['times'], y_fit, 'r', label='Fit (GNSS)', linewidth=2)

    W = np.linalg.inv(model_gnss['Sigma_Y'])
    ss_res = (model_gnss['y'] - y_fit).T @ W @ (model_gnss['y'] - y_fit)
    plt.title(f'Mean of squared residuals: {ss_res:.0f}')
    plt.grid()
    plt.legend()
    plt.show()

x0_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.1, description='x0')
x1_slider = widgets.FloatSlider(value=0, min=-0.1, max=0.1, step=0.001, description='x1')
x2_slider = widgets.FloatSlider(value=0, min=-1, max=1, step=0.01, description='x2')

interact(update_plot, x0=x0_slider, x1=x1_slider, x2=x2_slider)

xhat_slider_plot(model_gnss['A'], model_gnss['y'], model_gnss['times'], model_gnss['Sigma_Y'])

