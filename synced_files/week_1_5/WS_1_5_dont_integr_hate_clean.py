# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams['figure.figsize'] = (15, 5)  
plt.rcParams.update({'font.size': 13})    

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

def f(x):
    return 20*np.cos(x)+3*x**2

# %% [markdown]

# %%
f_at_x_equal_0 = YOUR_CODE_HERE

print("f evaluated at x=0 is:" , f_at_x_equal_0)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
a = YOUR_CODE_HERE
b = YOUR_CODE_HERE
number_of_points = YOUR_CODE_HERE

x_values = np.linspace(YOUR_CODE_HERE)
dx = x_values[1]-x_values[0]

print("x  = ",x_values)

assert abs(dx - np.pi)<1e-5, "Oops! dx is not equal to pi. Please check your values for a, b and number of points."

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

x_high_resolution = np.linspace(a, b, 50)

f_high_resolution = [ f(x) for x in x_high_resolution ] 

f_high_resolution = [ f(x_high_resolution[i]) for i in range(len(x_high_resolution))] 

plt.plot(x_high_resolution, f_high_resolution, '+', markersize='12', color='black')
plt.plot(x_high_resolution, f_high_resolution, 'b')
plt.legend(['Points evaluated','Continuous function representation'])
plt.title('Function for approximation')
plt.xlabel('x')
plt.ylabel('$f(x)$');

# %% [markdown]

# %% [markdown]

# %%

x_values = np.linspace(a, b, 10) 
dx = x_values[1]-x_values[0]

I_left_riemann = sum( [f(x)*dx for x in x_values[:-1]] )  
print(f"Left Riemann Sum: {I_left_riemann: 0.3f}")

I_left_riemann = sum( [ f(x_values[i])*dx for i in range(len(x_values)-1) ] )  
print(f"Left Riemann Sum: {I_left_riemann: 0.3f}")

# %% [markdown]

# %%

plt.bar(x_values[:-1],[f(x) for x in x_values[:-1]], width=dx, alpha=0.5, align='edge', edgecolor='black', linewidth=0.25)
plt.plot(x_values[:-1],[f(x) for x in x_values[:-1]], '*', markersize='16', color='red')

plt.plot(x_high_resolution, f_high_resolution, 'b')
plt.title('Left Riemann Sum')
plt.xlabel('x')
plt.ylabel('$f(x)$');

# %% [markdown]

# %% [markdown]

# %%
I_right_riemann = sum( [YOUR_CODE_HERE for x in YOUR_CODE_HERE] ) 

print(f"Right Riemann Sum: {I_right_riemann: 0.3f}")

# %% [markdown]

# %%

plt.bar(x_values[YOUR_CODE_HERE],[f(x) for x in x_values[YOUR_CODE_HERE]],
        width=YOUR_CODE_HERE, alpha=0.5, align='edge',
        edgecolor='black', linewidth=0.25)
plt.plot(x_values[YOUR_CODE_HERE],[f(x) for x in x_values[YOUR_CODE_HERE]],
         '*', markersize='16', color='red')

plt.plot(x_high_resolution, f_high_resolution, 'b')
plt.title('Right Riemann Sum')
plt.xlabel('x')
plt.ylabel('$f(x)$');

# %% [markdown]

# %% [markdown]

# %%
I_midpoint = sum([f(YOUR_CODE_HERE)*dx for i in range(YOUR_CODE_HERE)])
print(f"Midpoint Sum: {I_midpoint: 0.3e}")

I_midpoint = sum([f(x_at_the_middle)*dx for x_at_the_middle in YOUR_CODE_HERE ])
print(f"Midpoint Sum: {I_midpoint: 0.3e}")

# %% [markdown]

# %%

plt.bar(x_values[YOUR_CODE_HERE],[f(x_at_the_middle) for x_at_the_middle in YOUR_CODE_HERE ], width=dx, alpha=0.5, align='edge', edgecolor='black', linewidth=0.25)
plt.plot(x_values[YOUR_CODE_HERE],[f(x_at_the_middle) for x_at_the_middle in YOUR_CODE_HERE],'*',markersize='16', color='red')

plt.plot(x_high_resolution, f_high_resolution, 'b')
plt.title('Midpoint Sum')
plt.xlabel('x')
plt.ylabel('$f(x)$');

# %% [markdown]

# %% [markdown]

# %%
I_trapezoidal = sum([YOUR_CODE_HERE for i in range(len(x_values)-1)]) 
print(f"Trapezoidal Sum: {I_trapezoidal: 0.5e}")

# %% [markdown]

# %%

for i in range(len(x_values)-1):
    plt.fill_between([x_values[i], x_values[i+1]], 
                     [f(x_values[i]), f(x_values[i+1])], 
                     alpha=0.5)

plt.plot(x_high_resolution, f_high_resolution, 'b')
plt.title('Trapezoidal Sum')
plt.xlabel('x')
plt.ylabel('$f(x)$');

# %% [markdown]

# %% [markdown]

# %%

left_riemann_error = YOUR_CODE_HERE
right_riemann_error =  YOUR_CODE_HERE
midpoint_error = YOUR_CODE_HERE
trapezoidal_error = YOUR_CODE_HERE

print(f"Left Riemann Error: {left_riemann_error: 0.3e}")
print(f"Right Riemann Error: {right_riemann_error: 0.3e}")
print(f"Midpoint Error: {midpoint_error: 0.3e}")
print(f"Trapezoidal Error: {trapezoidal_error: 0.3e}")

# %% [markdown]

# %% [markdown]

# %%

x_values = np.linspace(YOUR_CODE_HERE)
dx = YOUR_CODE_HERE                      
    
simpson_integral = sum([ YOUR_CODE_HERE ]) 

simpson_error = YOUR_CODE_HERE

print(f"Simpson's Rule Integral: {simpson_integral: 0.5e}")
print(f"Simpson's Rule Absolute Error: {simpson_error: 0.5e}")

# %% [markdown]

# %% [markdown]

# %% [markdown]

