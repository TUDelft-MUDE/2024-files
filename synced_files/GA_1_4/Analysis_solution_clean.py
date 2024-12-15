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

print(m1['Sigma_Y'])
print(m2['Sigma_Y'])

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
    
    # Hint: use d['days'] and d['groundwater'] for
    #       the deterministic parameters

    # y_comp = (YOUR_CODE_HERE
    #           + YOUR_CODE_HERE*(1 - np.exp(-d['days']/YOUR_CODE_HERE))
    #           + YOUR_CODE_HEREX[3]*d['groundwater']
    #      )
    
    # SOLUTION:
    y_comp = (x[0]
         + x[1]*(1 - np.exp(-d['days']/x[2]))
         + x[3]*d['groundwater']
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

d_init = 9
R_init = -25
a_init = 300
k_init = 0.15

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
    # Hint: use d['days'] and d['groundwater'] for
    #       the deterministic parameters

    # J1 = 
    # J2 = 
    # J3 = 
    # J4 = 
    # J = YOUR_CODE_HERE

    # SOLUTION
    J1 = np.ones(len(d['days']))                              
    J2 = 1 - np.exp(-d['days']/x[2])                          
    J3 = -x[1] * d['days']/x[2]**2 * np.exp(-d['days']/x[2])  
    J4 = np.ones(len(d['days']))*d['groundwater']             
    J = np.column_stack((J1, J2, J3, J4))
    
    return J

test_J = jacobian((d_init, R_init, a_init, k_init), m1)

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

    # Define observations as an array
    # (we will overwrite it temporarily)
    y_obs = d['y']

    while x_norm >= 1e-12 and iteration < 49:

        # y_comp_i = YOUR_CODE_HERE
        # Delta_y = YOUR_CODE_HERE
        # J_i = YOUR_CODE_HERE

        # SOLUTION
        y_comp_i = compute_y(x_hat_i[iteration, :], d)
        Delta_y = y_obs - y_comp_i
        J_i = jacobian(x_hat_i[iteration, :], d)

        # d[YOUR_CODE_HERE] = YOUR_CODE_HERE
        # d[YOUR_CODE_HERE] = YOUR_CODE_HERE
        # Hints for previous line:
        #   - re-use your function BLUE
        #   - you will need to repurpose two dictionary
        #     keys to utilize the solution scheme of BLUE
        #     that can solve linear equations

        # SOLUTION
        d['y'] = Delta_y
        d['A'] = J_i

        d = BLUE(d)
        
        # X_hat_i[iteration+1,:] = YOUR_CODE_HERE
        # Hints for previous line:
        #   - now repurpose a result stored in the dictionary

        # SOLUTION
        x_hat_i[iteration+1,:] = x_hat_i[iteration,:] + d['X_hat'].T
        
        # x_norm = YOUR_CODE_HERE
        
        # SOLUTION
        x_norm = d['X_hat'].T @ np.linalg.inv(d['Sigma_X_hat']) @ d['X_hat']

        # Update the iteration number
        iteration += 1

        if iteration==49:
            print("Number of iterations too large, check initial values.")

    # # Store general results from the iterative process
    # d['x_hat_all_iterations'] = YOUR_CODE_HERE
    # d['iterations_completed'] = YOUR_CODE_HERE

    # # Store the linear values and "Reset" the non-linear ones
    # # Two sets of values correspond to Y and X
    # d['Delta_y'] = YOUR_CODE_HERE
    # d['y'] = YOUR_CODE_HERE
    
    # d['Delta_x'] = YOUR_CODE_HERE
    # d['X_hat'] = YOUR_CODE_HERE

    # SOLUTION
    # Store general results from the iterative process
    d['x_hat_all_iterations'] = x_hat_i[0:iteration+1, :]
    d['iterations_completed'] = iteration

    # Store the linear values and "Reset" the non-linear ones
    # Two sets of values correspond to Y and X
    d['Delta_y'] = d['y']
    d['y'] = y_obs
    
    d['Delta_x'] = d['X_hat']
    d['X_hat'] = d['x_hat_all_iterations'][iteration,:]
    
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
    plt.plot(d['x_hat_all_iterations'][:,0], linewidth=4)
    plt.title('Estimated offset')
    plt.ylabel('Offset [mm]')
    plt.xlabel('Number of iterations [-]')

    plt.subplot(2,2,2)
    plt.plot(d['x_hat_all_iterations'][:,1], linewidth=4)
    plt.title('Estimated R value')
    plt.ylabel('Estimated R value [mm]')
    plt.xlabel('Number of iterations [-]')

    plt.subplot(2,2,3)
    plt.plot(d['x_hat_all_iterations'][:,2], linewidth=4)
    plt.title('Estimated $a$ value')
    plt.ylabel('a value [days]')
    plt.xlabel('Number of iterations [-]')

    plt.subplot(2,2,4)
    plt.plot(d['x_hat_all_iterations'][:,3], linewidth=4)
    plt.title('Estimated GW factor')
    plt.ylabel('Estimated GW factor [-]')
    plt.xlabel('Number of iterations [-]')

plot_fit_iteration(m1)
plot_fit_iteration(m2)

initial_guess_alternative = initial_guess
print(initial_guess_alternative)
plot_convergence_interactive(gauss_newton_iteration(initial_guess_alternative, m1))
plot_convergence_interactive(gauss_newton_iteration(initial_guess_alternative, m2))

    

def show_std(Sigma_X_hat, data_type):
    print ('The standard deviation for',
           data_type + '-offset is',
           np.round(np.sqrt(Sigma_X_hat[0,0]),2), 'mm')
    print ('The standard deviation for',
           data_type + '-R is',
           np.round(np.sqrt(Sigma_X_hat[1,1]),2), 'mm')
    print ('The standard deviation for',
           data_type + '-a is',
           np.round(np.sqrt(Sigma_X_hat[2,2]),2), 'days')
    print ('The standard deviation for',
           data_type + '-the ground water factor',
           np.round(np.sqrt(Sigma_X_hat[3,3]),3), '[-]')
    

print ('Covariance matrix of estimated parameters (InSAR):')
print (m1['Sigma_X_hat'], '\n')
show_std(m1['Sigma_X_hat'], 'InSAR')
print()
model_summary(m1)

print ('Covariance matrix of estimated parameters (GNSS):')
print (m2['Sigma_X_hat'], '\n')
show_std(m2['Sigma_X_hat'], 'GNSS')
print()
model_summary(m2)

help(get_CI)

m1['Y_hat'] = compute_y(m1['X_hat'], m1)
m1 = get_CI(m1, 0.04)
plot_model(m1)
plot_residual(m1)
plot_residual_histogram(m1);

m2['Y_hat'] = compute_y(m2['X_hat'], m2)
m2 = get_CI(m2, 0.04)
plot_model(m2)
plot_residual(m2)
plot_residual_histogram(m2);

q = 1
alpha = 0.005 
k = chi2.ppf(1 - alpha, df=q)
print(f'The critical value is {np.round(k, 3)}')

t1_insar = (m1_blue['e_hat'].T
            @ np.linalg.inv(m1_blue['Sigma_Y'])
            @ m1_blue['e_hat'])
t2_insar = (m1['e_hat'].T
            @ np.linalg.inv(m1['Sigma_Y'])
            @ m1['e_hat'])
t_insar = t1_insar - t2_insar
print(f'The test statistic for InSAR data is {np.round(t_insar, 3)}')

t1_gnss = (m2_blue['e_hat'].T
           @np.linalg.inv(m2_blue['Sigma_Y'])
           @m2_blue['e_hat'])
t2_gnss = (m2['e_hat'].T
           @np.linalg.inv(m2['Sigma_Y'])
           @m2['e_hat'])
t_gnss = t1_gnss - t2_gnss
print(f'The test statistic for GNSS data is {np.round(t_gnss, 3)}')

