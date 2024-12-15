import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

def g(x):
    return YOUR_CODE_HERE

def g_der(x):
    return YOUR_CODE_HERE

x = YOUR_CODE_HERE
for j in range(100):
    x = YOUR_CODE_HERE
    # Next task will go here

print("The solution found is ", x, " it took " ,j , " iterations to converge.")

def g(y_iplus1, y_i, t_iplus1):
    return YOUR_CODE_HERE

def g_der(y_iplus1):
    return YOUR_CODE_HERE

dt = .25
t_end = 10
t = np.arange(0,t_end+dt,dt)

y_EE = np.zeros(t.shape)
y_IE = np.zeros(t.shape)

y_EE[0] = YOUR_CODE_HERE
y_IE[0] = YOUR_CODE_HERE

newtonFailed = 0
for i in range(0, len(t)-1):    
    
    # Forward Euler:
    y_EE[i+1] = YOUR_CODE_HERE

    # Backward Euler:
    y_IE[i+1] = YOUR_CODE_HERE # Initial guess
    for j in range(200):
        y_IE[i+1] = YOUR_CODE_HERE
        if np.abs(g(y_IE[i+1], y_IE[i], t[i+1])) < 1e-6:
            break
        
    if j >= 199:
        newtonFailed = 1
    

plt.plot(t, y_EE, 'r', t, y_IE, 'g--')
if newtonFailed:
    plt.title('Nonlinear ODE with dt = ' + str(dt) + ' \nImplicit Euler did not converge')
else:
    plt.title('Nonlinear ODE with dt = ' + str(dt))

plt.xlabel('t')
plt.ylabel('y')
plt.gca().legend(('Explicit','Implicit'))
plt.grid()
plt.show()

T_left = YOUR_CODE_HERE # Temperature at left boundary
T_right = YOUR_CODE_HERE # Temperature at right boundary
T_initial = YOUR_CODE_HERE # Initial temperature

length = YOUR_CODE_HERE # Length of the rod
nu = YOUR_CODE_HERE # Thermal diffusivity

dx = YOUR_CODE_HERE # spatial step size
x = YOUR_CODE_HERE # spatial grid

dt = YOUR_CODE_HERE # time step size

T = YOUR_CODE_HERE
T[0, :] = YOUR_CODE_HERE
T[:, 0] = YOUR_CODE_HERE
T[:, -1] = YOUR_CODE_HERE

b = YOUR_CODE_HERE

for j in range(m-1):
    A = YOUR_CODE_HERE
    b = YOUR_CODE_HERE
    
    T[j+1,1:-1] = YOUR_CODE_HERE

def plot_T(T):
    '''
    Function to plot the temperature profile at different time steps.
    '''
    def plot_temperature(time_step):
        plt.plot(x, T[time_step, :])
        plt.xlabel('x [m]')
        plt.ylabel('T [°C]')
        plt.title(f'Temperature profile at time step {time_step}')
        plt.grid()
        plt.ylim(5, 40)
        plt.show()

    interact(plot_temperature, time_step=widgets.Play(min=0, max=len(t)-1, step=3, value=0))

plot_T(T)

YOUR_CODE_HERE

YOUR_CODE_HERE

