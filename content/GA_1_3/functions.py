
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    y_hat = d['y_hat']
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
                    label = f'{(1-d['alpha'])*100:.1f}%' + ' Confidence Region')
    
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
    CI_res = d['CI_res']
    data_type = d['data_type']

    fig, ax = plt.subplots(figsize = (15,5))

    ax.plot(times, e_hat, 'o', markeredgecolor='black', label='Residual')
    ax.plot(times, -CI_res,
             '--', color='orange', 
             label = f'{(1-d['alpha'])*100:.1f}%' + ' Confidence Region')
    ax.plot(times, +CI_res,
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
    
    ax.hist(e_hat, bins = 40, density=True,  edgecolor='black')
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
