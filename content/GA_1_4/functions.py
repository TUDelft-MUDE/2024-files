
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
    CI_res = d['CI_res']
    data_type = d['data_type']

    fig, ax = plt.subplots(figsize = (15,5))

    ax.plot(times, e_hat, 'o', markeredgecolor='black', label='Residual')
    ax.plot(times, -CI_res,
             '--', color='orange', 
             label = f'{(1-d["alpha"])*100:.1f}%' + ' Confidence Region')
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

def xhat_slider_plot(A, y, t, Sigma_y=None):
    """Interactive plot of the solution space for x_hat."""
    n, k = A.shape

    if Sigma_y is None:
        Sigma_y = np.eye(n)
    xhat = np.linalg.inv(A.T @ np.linalg.inv(Sigma_y) @ A) @ A.T @ np.linalg.inv(Sigma_y) @ y
    Sigma_xhat = np.linalg.inv(A.T @ np.linalg.inv(Sigma_y) @ A)
    std_xhat = np.sqrt(np.diag(Sigma_xhat))

    sliders = {}
    for i in range(k):
        sliders[f'xhat_{i}'] = widgets.FloatSlider(value=xhat[i],
                                                   min=xhat[i] - 10*std_xhat[i],
                                                   max=xhat[i] + 10*std_xhat[i],
                                                   step=0.1*std_xhat[i],
                                                   description=f'xhat_{i}')

    def update_plot(**kwargs):
        xhat_values = np.array([kwargs[f'xhat_{i}'] for i in range(k)])
        y_fit = A @ xhat_values
        W = np.linalg.inv(Sigma_y)
        ss_res = (y - y_fit).T @ W @ (y - y_fit)

        plt.figure(figsize=(10, 5))
        plt.plot(t, y, 'o', label='data')
        plt.plot(t, y_fit, label='y_fit')
        plt.title(f'Mean of squared residuals: {ss_res:.2f}')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.grid()
        plt.legend()
        plt.show()

    interact(update_plot, **sliders)

# Example usage
# A, y, t should be defined before calling this function
# xhat_slider_plot(A, y, t)

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
    d['x_hat'] = (d['Sigma_X_hat']
                  @ d['A'].T
                  @ np.linalg.inv(d['Sigma_Y'])
                  @ d['y'])
    d['y_hat'] = d['A'] @ d['x_hat']
    d['e_hat'] = d['y'] - d['y_hat']
    d['Sigma_Y_hat'] = d['A'] @ d['Sigma_X_hat'] @ d['A'].T
    d['std_y'] = np.sqrt(d['Sigma_Y_hat'].diagonal())
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
    d['CI_y'] = d['k']*d['std_y']
    d['CI_res'] = d['k']*d['std_e_hat']
    d['CI_y_hat'] = d['k']*np.sqrt(d['Sigma_Y_hat'].diagonal())

    return d

def model_summary(d):
    """Print the model summary."""

    print('Summary of Model')
    print('----------------')
    print('  Data type:', d['data_type'])
    print('  Model type:', d['model_type'])
    print('  Number of observations:', len(d['times']))
    print('  Model parameters:')
    for i in range(len(d['x_hat'])):
        print(f'    x_{i} = {d["x_hat"][i]:6.3f}'
              + f'  +/- {np.sqrt(d["Sigma_X_hat"][i,i]):6.3f}')
    print('----------------\n')