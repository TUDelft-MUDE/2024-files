import numpy as np
from scipy import interpolate
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, fixed
from scipy.stats.distributions import chi2
from functions import *
np.set_printoptions(precision=3);
m1_blue = load_pickle_file('m1_blue.pickle')
m2_blue = load_pickle_file('m2_blue.pickle')
model_summary(m1_blue)
model_summary(m2_blue)
for key in m1_blue.keys():
    print(key)
x0_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.1, description='x0')
x1_slider = widgets.FloatSlider(value=0, min=-0.1, max=0.1, step=0.001, description='x1')
x2_slider = widgets.FloatSlider(value=1, min=-1, max=1, step=0.01, description='x2')
interact(model_widget,
         x0=x0_slider, x1=x1_slider, x2=x2_slider, x3=fixed(None),
         m=[('INSAR', m1_blue), ('GNSS', m2_blue)]);
def initialize_new_dict(d_old):
    d = {}
    d['data_type'] = d_old['data_type']
    d['model_type'] = 'Non-Linear Least Squares'
    d['times'] = d_old['times']
    d['y'] = d_old['y']
    d['std_Y'] = d_old['std_Y']
    d['Sigma_Y'] = d_old['Sigma_Y']
    d['days'] = d_old['days']
    d['groundwater'] = d_old['groundwater']
    d['groundwater_data'] = d_old['groundwater_data']
    return d
m1 = initialize_new_dict(m1_blue)
m2 = initialize_new_dict(m2_blue)
YOUR_CODE_HERE
def compute_y(x, d):
    """Model, q: ground surface displacement.
    Inputs:
      x: tuple, list or array of parameters
         (d, R, a, k)
      d: dictionary of model parameters
    Outputs: ndarray of ground level
    Note: "times" is not included as an argument because
    the model is only configured to compute the
    model at times where the groundwater measurements have
    been interpolated (i.e., the times of the satellite
    observations). To use this model for prediction, the
    interpolation function must be incorporated in the
    dictionary and/or the function. 
    """
    y_comp = (YOUR_CODE_HERE
              + YOUR_CODE_HERE*(1 - np.exp(-d['days']/YOUR_CODE_HERE))
              + YOUR_CODE_HEREX[3]*d['groundwater']
         )
    return y_comp
m1['compute_y'] = compute_y
m2['compute_y'] = compute_y
x0_slider = widgets.FloatSlider(value=0, min=-40, max=40, step=0.5, description='x0')
x1_slider = widgets.FloatSlider(value=0, min=-50, max=50, step=1, description='x1')
x2_slider = widgets.FloatSlider(value=1, min=10, max=1000, step=10, description='x2')
x3_slider = widgets.FloatSlider(value=1, min=-1, max=1, step=0.01, description='x3')
interact(model_widget,
         x0=x0_slider, x1=x1_slider, x2=x2_slider, x3=x3_slider,
         m=[('InSAR', m1), ('GNSS', m2)]);
d_init = YOUR_CODE_HERE
R_init = YOUR_CODE_HERE
a_init = YOUR_CODE_HERE
k_init = YOUR_CODE_HERE
initial_guess = (d_init, R_init, a_init, k_init)
def jacobian(x, d):
    """Compute Jacobian of the model.
    Model, q: ground surface displacement.
    Inputs:
      x: tuple, list or array of parameters
         (d, R, a, k)
      d: dictionary of model parameters
    Outputs: The Jacobian matrix, J
             - partial derivatives w.r.t. parameters)
             - J1 through J4: derivative w.r.t. d, R, a, k
    Note: "times" is not included as an argument because
    the model is only configured to compute the
    model at times where the groundwater measurements have
    been interpolated (i.e., the times of the satellite
    observations). To use this model for prediction, the
    interpolation function must be incorporated in the
    dictionary and/or the function. 
    """
    J1 = YOUR_CODE_HERE
    J2 = YOUR_CODE_HERE
    J3 = YOUR_CODE_HERE
    J4 = YOUR_CODE_HERE
    J = YOUR_CODE_HERE
    return J
YOUR_CODE_HERE
print ('The first 5 rows of the Jacobian matrix (InSAR):')
print (test_J[0:5,:])
n_2 = np.shape(test_J)[1]
print(f'\nThe number of unknowns is {n_2}')
print(f'The redundancy (InSAR) is {m1["y"].shape[0] - n_2}')
print(f'The redundancy (GNSS) is {m2["y"].shape[0] - n_2}')
def gauss_newton_iteration(x0, d):
    """Use Gauss-Newton iteration to find non-linear parameters.
    Inputs:
      x0: initial guess for the parameters (d, R, a, k)
      d: dictionary of model parameters
    Outputs: dictionary with the non-linear model results.
    """
    x_norm = 1000 # initialize stop criteria
    x_hat_i = np.zeros((50, 4))
    x_hat_i[0,:] = x0
    iteration = 0
    y_obs = d['y']
    while x_norm >= 1e-12 and iteration < 49:
        y_comp_i = YOUR_CODE_HERE
        Delta_y = YOUR_CODE_HERE
        J_i = YOUR_CODE_HERE
        d[YOUR_CODE_HERE] = YOUR_CODE_HERE
        d[YOUR_CODE_HERE] = YOUR_CODE_HERE
        d = BLUE(d)
        X_hat_i[iteration+1,:] = YOUR_CODE_HERE
        x_norm = YOUR_CODE_HERE
        iteration += 1
        if iteration==49:
            print("Number of iterations too large, check initial values.")
    d['x_hat_all_iterations'] = YOUR_CODE_HERE
    d['iterations_completed'] = YOUR_CODE_HERE
    d['Delta_y'] = YOUR_CODE_HERE
    d['y'] = YOUR_CODE_HERE
    d['Delta_x'] = YOUR_CODE_HERE
    d['X_hat'] = YOUR_CODE_HERE
    return d
m1 = gauss_newton_iteration(initial_guess, m1)
m2 = gauss_newton_iteration(initial_guess, m2)
print('\n InSAR Reults for each iteration (Iterations completed =',
      m1['iterations_completed'], ')')
print(m1['x_hat_all_iterations'])
print('\n GNSS Reults for each iteration (Iterations completed =',
      m2['iterations_completed'], ')')
print(m2['x_hat_all_iterations'])
def plot_fit_iteration(d):
    """Plot value of each parameter, each iteration."""
    plt.figure(figsize = (15,4))
    plt.subplots_adjust(top = 2)
    plt.subplot(2,2,1)
    YOUR_CODE_HERE
    plt.title('YOUR_CODE_HERE')
    plt.ylabel('YOUR_CODE_HERE [UNITS]')
    plt.xlabel('Number of iterations [-]')
    plt.subplot(2,2,2)
    YOUR_CODE_HERE
    plt.title('YOUR_CODE_HERE')
    plt.ylabel('YOUR_CODE_HERE [UNITS]')
    plt.xlabel('Number of iterations [-]')
    plt.subplot(2,2,3)
    YOUR_CODE_HERE
    plt.title('YOUR_CODE_HERE')
    plt.ylabel('YOUR_CODE_HERE [UNITS]')
    plt.xlabel('Number of iterations [-]')
    plt.subplot(2,2,4)
    YOUR_CODE_HERE
    plt.title('YOUR_CODE_HERE')
    plt.ylabel('YOUR_CODE_HERE [UNITS]')
    plt.xlabel('Number of iterations [-]')
plot_fit_iteration(m1)
plot_fit_iteration(m2)
def show_std(Sigma_X_hat, data_type):
    print ('The standard deviation for',
           data_type + '-offset is',
           YOUR_CODE_HERE, 'UNITS')
    print ('The standard deviation for',
           data_type + '-R is',
           YOUR_CODE_HERE, 'UNITS')
    print ('The standard deviation for',
           data_type + '-a is',
           YOUR_CODE_HERE, 'UNITS')
    print ('The standard deviation for',
           data_type + '-the ground water factor',
           YOUR_CODE_HERE, 'UNITS')
print ('Covariance matrix of estimated parameters (InSAR):')
print (YOUR_CODE_HERE, '\n')
show_std(YOUR_CODE_HERE)
print()
model_summary(YOUR_CODE_HERE)
print ('Covariance matrix of estimated parameters (GNSS):')
print (YOUR_CODE_HERE, '\n')
show_std(YOUR_CODE_HERE)
print()
model_summary(YOUR_CODE_HERE)
help(get_CI)
m1['Y_hat'] = YOUR_CODE_HERE
m1 = get_CI(YOUR_CODE_HERE)
plot_model(YOUR_CODE_HERE)
plot_residual(YOUR_CODE_HERE)
plot_residual_histogram(YOUR_CODE_HERE);
m1['Y_hat'] = YOUR_CODE_HERE
m1 = get_CI(YOUR_CODE_HERE)
plot_model(YOUR_CODE_HERE)
plot_residual(YOUR_CODE_HERE)
plot_residual_histogram(YOUR_CODE_HERE);
YOUR_CODE_HERE (probably will be more than one line)
print(f'The critical value is {np.round(k, 3)}')
t1_insar = YOUR_CODE_HERE
t2_insar = YOUR_CODE_HERE
t_insar = t1_insar - t2_insar
print(f'The test statistic for InSAR data is {np.round(t_insar, 3)}')
t1_gnss = YOUR_CODE_HERE
t2_gnss = YOUR_CODE_HERE
t_gnss = t1_gnss - t2_gnss
print(f'The test statistic for GNSS data is {np.round(t_gnss, 3)}')
