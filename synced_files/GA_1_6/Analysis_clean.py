# ---

# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def g(x):
    return YOUR_CODE_HERE

def g_der(x):
    return YOUR_CODE_HERE

x = YOUR_CODE_HERE
for j in range(100):
    x = YOUR_CODE_HERE
    

print("The solution found is ", x, " it took " ,j , " iterations to converge.")

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
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
    
    
    y_EE[i+1] = YOUR_CODE_HERE

    
    y_IE[i+1] = YOUR_CODE_HERE 
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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

T_left = YOUR_CODE_HERE 
T_right = YOUR_CODE_HERE 
T_initial = YOUR_CODE_HERE 

length = YOUR_CODE_HERE 
nu = YOUR_CODE_HERE 

dx = YOUR_CODE_HERE 
x = YOUR_CODE_HERE 

dt = YOUR_CODE_HERE 

# %% [markdown]

# %% [markdown]

# %%
T = YOUR_CODE_HERE
T[0, :] = YOUR_CODE_HERE
T[:, 0] = YOUR_CODE_HERE
T[:, -1] = YOUR_CODE_HERE

b = YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %%

for j in range(m-1):
    A = YOUR_CODE_HERE
    b = YOUR_CODE_HERE
    
    T[j+1,1:-1] = YOUR_CODE_HERE

# %% [markdown]

# %%
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

# %% [markdown]

# %%
plot_T(T)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

