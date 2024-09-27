import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import ipywidgets as widgets
from ipywidgets import interact

def plot_model(d, alt_model=None):
    """Time Series of observations with model and CI.
    
    Uses dict as input defined from existing values in dict
    Outputs figure and axes objects.

    alt_model: tuple to add a line to the plot, 3 elements:
      - string: array with the times
      - list or ndarray with times (x-axis)
      - list or ndarray with model values (y-axis)
      - e.g., alt_model=('model 2', [...], [...])
    """
    
    times = d['times']
    y = d['y']
    y_hat = d['Y_hat']
    CI_Y_hat = d['CI_Y_hat']
    data_type = d['data_type']

    fig, ax = plt.subplots(figsize = (15,5))

    ax.plot(times, y, 'k+',  label = 'Observations')
    ax.plot(times, y_hat,  label = 'Fitted model')
    ax.fill_between(times,
                    (y_hat - CI_Y_hat), 
                    (y_hat + CI_Y_hat),
                    facecolor='orange',
                    alpha=0.4,
                    label = f'{(1-d["alpha"])*100:.1f}%' + ' Confidence Region')
    
    if alt_model is not None:
        ax.plot(alt_model[1],
                alt_model[2],
                label = alt_model[0])
    
    ax.legend()
    ax.set(xlabel='Time',
           ylabel=f'{data_type}' + ' Displacement [mm]',
           title=f'{data_type}' + ' Observations and Fitted Model')

    
    return fig, ax


def plot_residual(d):
    """Time Series of residuals and CI.
    
    Uses dict as input defined from existing values in dict
    Outputs figure and axes objects.
    """
    
    times = d['times']
    e_hat = d['e_hat']
    CI_e_hat = d['CI_e_hat']
    data_type = d['data_type']

    fig, ax = plt.subplots(figsize = (15,5))

    ax.plot(times, e_hat, 'o', markeredgecolor='black', label='Residual')
    ax.plot(times, - CI_e_hat,
             '--', color='orange', 
             label = f'{(1-d["alpha"])*100:.1f}%' + ' Confidence Region')
    ax.plot(times, + CI_e_hat,
             '--', color='orange')
    
    ax.legend()
    ax.set(xlabel='Time',
           ylabel=f'{data_type}' + ' residual [mm]',
           title=f'{data_type}' + ' Residuals')

    return fig, ax


def plot_residual_histogram(d):
    """Histogram of residuals with Normal PDF.
    
    Uses dict as input defined from existing values in dict
    Outputs figure and axes objects.
    """
    
    e_hat = d['e_hat']
    data_type = d['data_type']

    fig, ax = plt.subplots(figsize = (7,5))
    
    ax.hist(e_hat, density=True,  edgecolor='black')
    x = np.linspace(np.min(e_hat), np.max(e_hat), num=100);
    ax.plot(x,
            norm.pdf(x, loc=0.0, scale = np.std(e_hat)),
            linewidth=4.0)

    ax.set(xlabel='Residuals [mm]',
           ylabel=f'{data_type}' + ' Density [-]',
           title=f'{data_type}' + ' Residuals Histogram')
    
    print ('The mean value of the', data_type, 'residuals is',
           np.around(np.mean(e_hat),5), 'mm')
    print ('The standard deviation of the', data_type, 'residuals is',
        np.around(np.std(e_hat),3), 'mm')


    return fig, ax

def model_widget(x0, x1, x2, x3, m):
    plt.figure(figsize=(15,5))
    plt.plot(m['times'], m['y'], 'o', label=m['data_type'])
    plt.ylabel('Displacement and Groundwater Elevation [mm]')
    plt.xlabel('Time')
    
    if x3 is None:
        y_fit = m['A'] @ [x0, x1, x2]
        if (x0 == 0) & (x1 == 0) & (x2 == 1):
            plt.plot(m['times'], y_fit, 'r', label='Groundwater Interpolation', linewidth=2)
        else:
            plt.plot(m['times'], y_fit, 'r', label='Fit', linewidth=2)
    else:
        compute_y = m['compute_y']
        x_hat = (x0, x1, x2, x3)
        y_fit = compute_y(x_hat, m)
        if (x0 == 0) & (x1 == 0) & (x2 == 10) & (x3 == 1):
            plt.plot(m['times'], y_fit, 'r', label='Groundwater Interpolation', linewidth=2)
        else:
            plt.plot(m['times'], y_fit, 'r', label='Fit', linewidth=2)
    
    plt.plot(m['groundwater_data']['times'],
             m['groundwater_data']['y'],
             'ro', label='Groundwater Data',
             markeredgecolor='black', markeredgewidth=1)

    if 'Sigma_Y' in m:
        W = np.linalg.inv(m['Sigma_Y'])
        ss_res = (m['y'] - y_fit).T @ W @ (m['y'] - y_fit)
        plt.title(f'Normalized squared residuals: {ss_res:.0f}')
    else:
        plt.title('Sigma_Y not yet defined in dictionary')
    
    plt.ylim(-150, 30)
    plt.grid()
    plt.legend()
    plt.show()

def to_days_years(times):
    '''Convert the observation times to days and years.'''
    
    times_datetime = pd.to_datetime(times)
    time_diff = (times_datetime - times_datetime[0])
    days_diff = (time_diff / np.timedelta64(1,'D')).astype(int)
    
    days = days_diff.to_numpy()
    years = days/365
    
    return days, years

def BLUE(d):
    """Calculate the Best Linear Unbiased Estimator
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    d['Sigma_X_hat'] = (np.linalg.inv(d['A'].T
                        @ np.linalg.inv(d['Sigma_Y'])
                        @ d['A']))
    d['X_hat'] = (d['Sigma_X_hat']
                  @ d['A'].T
                  @ np.linalg.inv(d['Sigma_Y'])
                  @ d['y'])
    d['Y_hat'] = d['A'] @ d['X_hat']
    d['e_hat'] = d['y'] - d['Y_hat']
    d['Sigma_Y_hat'] = d['A'] @ d['Sigma_X_hat'] @ d['A'].T
    d['Sigma_e_hat'] = d['Sigma_Y'] - d['Sigma_Y_hat']
    d['std_e_hat'] = np.sqrt(d['Sigma_e_hat'].diagonal())

    return d

def get_CI(d, alpha):
    """Compute the confidence intervals.
    
    Uses dict as input/output:
      - inputs defined from existing values in dict
      - outputs defined as new values in dict
    """

    d['k'] = norm.ppf(1 - 0.5*alpha)
    d['CI_Y'] = d['k']*d['std_Y']
    d['CI_e_hat'] = d['k']*d['std_e_hat']
    d['CI_Y_hat'] = d['k']*np.sqrt(d['Sigma_Y_hat'].diagonal())
    d['alpha'] = alpha

    return d

def model_summary(d):
    """Print the model summary."""

    print('Summary of Model')
    print('----------------')
    print('  Data type:', d['data_type'])
    print('  Model type:', d['model_type'])
    print('  Number of observations:', len(d['times']))
    print('  Model parameters:')
    for i in range(len(d['X_hat'])):
        print(f'    X_hat_{i} = {d["X_hat"][i]:8.3f}'
              + f'  +/- {np.sqrt(d["Sigma_X_hat"][i,i]):6.3f}'
              + f'  (c.o.v. '
              + f'{(np.sqrt(d["Sigma_X_hat"][i,i])/d["X_hat"][i]):6.3f})')
    print('----------------\n')

def load_pickle_file(filename):
    directory = os.path.join(os.path.dirname(__file__), 'auxiliary_files')
    filepath = os.path.join(directory, filename)
    # Load the data from the pickle file
    with open(os.path.normpath(filepath), 'rb') as file:
        data = pickle.load(file)     
    return data

def plot_model_specific_iteration(iteration, model):
    '''Plot the model at a specific iteration.
    
    Input:
        - iteration: integer
        - model: dictionary

    Used by widget plotting function:
        - plot_convergence_interactive

    Analogous to the plot_model function.
    '''
    x_hat_i = model['x_hat_all_iterations']
    times = model['times']
    y = model['y']
    y_hat = model['compute_y']((x_hat_i[iteration, :]), model)
    plt.figure(figsize=(16,4))
    plt.plot(times, y_hat , linewidth=4,
             label='Gauss Newton fit', color='black')
    plt.plot(times, y, 'co', mec='black',
             markersize=10, label='Observations',
             alpha=0.5)
    plt.legend()
    plt.xlabel('Time [days]')
    plt.ylabel('Water level [m]')
    plt.title(f'Iteration = {iteration}')
    plt.grid()
    plt.show()

def plot_parameter_convergence(iteration, x_hat_i):
    '''Plot the convergence of the parameters.
    
    Intput:
        - iteration: integer
        - x_hat_i: widget values, x_hat at specific iteration
    '''
    params = [f'x_{i}' for i in range(x_hat_i.shape[1])]
    
    fig, ax = plt.subplots(1,len(params), figsize=(16,4))
    for i in range(len(params)):
        ax[i].plot(x_hat_i[:, i].T, linewidth=4)
        ax[i].set_title(f'Convergence of {params[i]}')
        ax[i].set_xlabel(f'Number of iterations')
        ax[i].set_ylabel(f'{params[i]}')

    for i in range(len(params)):
        ax[i].plot(iteration, x_hat_i[iteration, i], 'ro')
    plt.show()

def plot_convergence_interactive(model):
    '''Interactive plot of the non-linear least squares iterations.
    
    Input: model dictionary

    Uses two special functions:
        - plot_model_specific_iteration
        - plot_parameter_convergence
    '''
    iteration = model['iterations_completed']
    y = model['y']
    i = widgets.IntSlider(min=0, max=iteration, step=1, value=0, interval=1000)
    interact(plot_model_specific_iteration,
             iteration=i, model=widgets.fixed(model));    
    interact(plot_parameter_convergence,
             iteration=i, x_hat_i=widgets.fixed(model['x_hat_all_iterations']));

