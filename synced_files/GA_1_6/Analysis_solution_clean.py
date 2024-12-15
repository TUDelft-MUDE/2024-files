import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

def g(x):
    return x**2 - 9

def g_der(x):
    return 2*x

x = 10
for j in range(100):
    x = x - g(x)/g_der(x)
    if np.abs(g(x)) < 1e-6:
        break
    
print("The solution found is ", x, " it took " ,j , " iterations to converge.")

x = .01
for j in range(100):
    x = x - g(x)/g_der(x)
    if np.abs(g(x)) < 1e-6:
        break
    
print("The solution found is ", x, " it took " ,j , " iterations to converge.")

    

        
    

def g(y_iplus1,y_i, t_iplus1):
    return y_iplus1-y_i-dt*(np.sin(y_iplus1**3)+np.sin(t_iplus1))

def g_der(y_iplus1):
    return 1-3*dt*y_iplus1**2*np.cos(y_iplus1**3)

dt = .25
t_end = 10
t = np.arange(0,t_end+dt,dt)

y_EE = np.zeros(t.shape)
y_IE = np.zeros(t.shape)

y_EE[0] = 1
y_IE[0] = 1

newtonFailed = 0
for i in range(0, len(t)-1):    
    
    # Forward Euler:
    y_EE[i+1] = y_EE[i] + dt*(np.sin(y_EE[i]**3)+np.sin(t[i]))

    # Backward Euler:
    y_IE[i+1] = y_IE[i] # initial guess
    for j in range(200):
        y_IE[i+1] = y_IE[i+1] - g(y_IE[i+1], y_IE[i], t[i+1]) / g_der(y_IE[i+1])
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

T_left = 38
T_right = 25
T_initial = 7
L = 0.3
nu = 4/1000/1000

dx = 0.02
x = np.arange(0,L,dx)
n = len(x)
dt = 50
m = 200

T = np.zeros((m,n))
T[0, :] = T_initial
T[:, 0] = T_left
T[:, -1] = T_right
b = T[j, 1:-1] + nu*dt/(dx**2)*(T[j, 2:]-2*T[j, 1:-1]+T[j, :-2])

for j in range(m-1):
    A = np.zeros((len(x)-2,len(x)-2))
    np.fill_diagonal(A, 1)
    b = T[j,1:-1] + nu*dt/(dx**2)*(T[j,2:]-2*T[j,1:-1]+T[j,:-2])    
    T_1_to_n_minus1 = np.linalg.inv(A) @ b
    T[j+1,1:-1] = T_1_to_n_minus1

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

    interact(plot_temperature, time_step=widgets.Play(min=0, max=len(T)-1, step=3, value=0))

plot_T(T)

T_left = 38
T_right = 25
T_initial = 7
L = 0.3
nu = 4/1000/1000

dx = 0.02
x = np.arange(0,L,dx)
n = len(x)
dt = 50
m = 200

period = 6000
T = np.zeros((m,n))
T[0,:] = T_initial
T[:,0] = T_left
t = np.arange(0,m*dt,dt)
T[:,-1] = 25 + 10*np.sin(2*np.pi*t/period)

for j in range(m-1):
    # Building matrix A
    A = np.zeros((len(x)-2,len(x)-2))
    np.fill_diagonal(A, 1)
    # Building vector b
    b = T[j,1:-1] + nu*dt/(dx**2)*(T[j,2:]-2*T[j,1:-1]+T[j,:-2])    
    T_1_to_n_minus1 = np.linalg.inv(A) @ b
    T[j+1,1:-1] = T_1_to_n_minus1

plot_T(T)

T_left = 38
T_right = 25
T_initial = 7
L = 0.3
nu = 4/1000/1000

dx = 0.02
x = np.arange(0,L,dx)
n = len(x)
dt = 50
m = 200

T = np.zeros((m,n))
T[0,:] = T_initial
T[:,0] = T_left
t = np.arange(0,m*dt,dt)
T[:,-1] = T_right

C = nu*dt/dx**2
for j in range(m-1):
    # Building matrix A
    A = np.zeros((len(x)-2,len(x)-2))
    np.fill_diagonal(A, 1+2*C)
    A[np.arange(n-3), np.arange(1, n-2)] = -C  # Upper diagonal
    A[np.arange(1, n-2), np.arange(n-3)] = -C  # Lower diagonal
    # Building vector b
    b = T[j,1:-1].copy()
    b[0] = b[0] + T_left * C
    b[-1] = b[-1] + T_right * C 
    T_1_to_n_minus1 = np.linalg.inv(A) @ b
    T[j+1,1:-1] = T_1_to_n_minus1

plot_T(T)

