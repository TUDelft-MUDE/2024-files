<userStyle>Normal</userStyle>

# WS 2.1: Wiggles

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 90px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 25px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1, Wednesday November 13, 2024.*


<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>General comments to start off the solution:</b>
</p>
</div>


1. This is in general how the code works: you should have seen that for each case, once you define the Python variables, there are ready-made functions to complete the analysis: first by running the simulation (`run_simulation_1D`) which initializes the volumes (`initialize_1D`) then sets up the time (the loop over `t`) and space integration (`advection_1D`). Then you can plot the initial condition (`plot_1D`) and visualize each time step (`plot_1D_all`).  
2. Applying the cyclic boundary condition is very simple: you only need to shift the indexing from the "math" in the textbook by 1 when implementing in Python. This does introduce a slight error since you are actually calculating $phi$ for a volume to the left of your index; however, for small volumes it is negligible (and you could shift it back in each time step, if you wanted).  
3. Note also that the cyclic boundary does not have any mathematical basis, and is not modelling "reality" in any way. It is a convenient "trick" that we use to make it easy to observe how the wave moves in space without worrying about the particular location of the wave (x) at any time.

Key things to notice about the activities in this assignment:  

1. In Part 1, we see that the central scheme is _very_ unstable (default settings).  
2. In Part 2: making the initial condition smooth helps...for a short time (set `square=False`).
3. In Part 3: upwinding becomes (conditionally) stable; if CFL<1.0, you will see false diffusion.
4. In Part 4: we use the CFL number to confirm stability regions. Note that if you use a CFL of 0.8 and your solution is unstable, you implemented the upwinding scheme incorrectly. Check the tips above about the cyclic indexing to help solve your issue; note also that regardless of if or how you implemented the cyclic boundary condition, if you are using points that are to the "right" of the index for `p_new` then you did not implement upwinding correctly (for example, if you compute `p_new[i]` and the calculation includes a point `p[i + 1]`).

The reason we notice the above points is because our PDE is convection (hyperbolic) and our pulse is moving to the right, which implies that the upwind cells (volumes) are more important (they have a larger weight). Even though backward difference (the upwind scheme) is a lower-accuracy method (first order) than central averaging (second order), it is more stable.

It is important to recognize **three types of numerical issues** that are present in this assignment:
1. Instability caused by the nature of the PDE (a hyperbolic equation), combined with the numerical scheme (central averaging in space and forward Euler in time).
2. Instability as described by the CFL criterion.
3. Numerical diffusion (false diffusion), which is inherent in upwind schemes (an explanation is outside of the scope of MUDE: we are happy as long as you recognize this can be an issue!).

_After the general comments here, the solution has been implemented in the code below. There is also a large section of code that facilitates the use of dictionaries to keep track of analysis cases; it's not part of the MUDE exam material, but you might find it useful when working on your projects, especially if you are working on a situation when you would like to evaluate a lot of scenarios and keep track of the input parameters. For example, you can see that each dictionary stores the variable values and a name describing it._


<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>End general comments.</b>
</p>
</div>

<!-- #region -->
## Overview:

In this workshop the [advection problem covered in the textbook] is treated first in 1D and then in 2D. There are two main objectives:
1. Understand the advection problem itself (how the quantity of interest is transported by the velocity field)
2. Explore characteristics of the numerical analysis schemes employed, in particular: numerical diffusion and FVM stability

To do this we will do the following:
- Implement the central difference and upwind schemes in space for FVM and Forward Euler in time
- Apply a boundary condition such that the quantity of interest repeatedly travels through the plot window (this helps us visualize the process!)
- Evaluate stability of central difference and upwind schemes in combination with Forward Euler in time
- Use the CFL to understand numerical stability

Programming requirements: you will need to fill in a few missing pieces of the functions, but mostly you will change the values of a few Python variables to evaluate different aspects of the problem.


The following Python variables will be defined to set up the problem:
```
p0 = initial value of our "pulse" (the quantity of interest, phi) [-]
c = speed of the velocity field [m/s]
L = length of the domain [m]
Nx = number of volumes in the direction x
T = duration of the simulation (maximum time) [s]
Nt = number of time steps
```

There are also two flag variables: 1) `central` will allow you to switch between central and backward spatial discretization schemes, and 2) `square` changes the pulse from a square to a smooth bell curve (default for both is `True`; don't worry about it until instructed to change it).

For the 2D case, `c`, `L`, `Nx` are extended as follows:
```
c --> cx, cy
L --> Lx, Ly
Nx -> Nx, Ny
```
<!-- #endregion -->

## Part 1: Implement Central Difference

We are going to implement the central difference scheme as derived in the textbook; however, **instead of implementing the system of equations in a matrix formulation**, we will _loop over each of the finite volumes in the system,_ one at a time.

Because we will want to watch the "pulse" travel over a long period of time, we will take advantage of the reverse-indexing of Python (i.e., the fact that an index `a[-3]`, for example, will access the third item from the end of the array or list). When taking the volumes to the left of the first volume in direction $x$, we can use the values of $phi$ from the "last" volumes in $x$ (the end of the array). All we need to do is shift the index for $phi_i$ such that we avoid a situation where `i+1` "breaks" the loop (because the maximum index is `i`). In other words, only volumes with index `i` or smaller should be used (e.g., instead of `i-1`, `i`,  and `i+1`, use `i-2`, `i-1` and `i`.



<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1:</b>

Write by hand for FMV the <code>advection_1D</code> equation, compute the convective fluxes of $\phi$ at the surfaces using a linear interpolation (central averaging). Then apply Forward Euler in time to the resulting ODE. Make sure you use the right indexing (maximum index should be <code>i</code>).

$$
\frac{\partial \phi}{\partial t} + c\frac{\partial \phi}{\partial x} = 0
$$
    
</div>



<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2:</b>

<b>Read.</b> Read the code to understand the problem that has been set up for you. Check the arguments and return values; the docstrings are purposefully ommitted so you can focus on the code. You might as well re-read the instructions above one more time, as well (let's be honest, you probably just skimmed over it anyway...)
    
</div>

```python
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from ipywidgets import interact, fixed, widgets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3:</b>

Implement the scheme in the function <code>advection_1D</code>. Make sure you use the right indexing (maximum index should be <code>i</code>).
    
</div>


Complete the functions to solve the 1D problem. A plotting function has already been defined, which will be used to check your initial conditions and visualize time steps in the solution.

```python
def initialize_1D(p0, L, Nx, T, Nt, square=True):
    """Initialize 1D advection simulation.

    Arguments are defined elsewhere, except one keyword argument
    defines the shape of the initial condition.
    
    square : bool
      - specifies a square pulse if True
      - specifies a Gaussian shape if False
    """

    dx = L/Nx
    dt = T/Nt
    
    x = np.linspace(dx/2, L - dx/2, Nx)

    # Initialise domain:
    #  - a square pulse with p0 between 0.5 and 1
    #  - a Gaussian shape around 1
    if square:
        p_init = np.zeros(Nx)
        p_init[int(.5/dx):int(1/dx + 1)] = p0
    else:
        p_init = np.exp(-((x-1.0)/0.5**2)**2)

    p_all = np.zeros((Nt+1, Nx))
    p_all[0] = p_init
    return x, p_all

def advection_1D(p, dx, dt, c, Nx, central=True):
    """Solve the advection problem."""
    p_new = np.zeros(Nx)
    for i in range(0, Nx):
        if central:
            # pass # add central averaging + FE scheme here (remove pass)
            p_new[i-1] = p[i - 1] - 0.5*(c*dt/dx)*(p[i] - p[i - 2])
        else:
            # pass # add upwind + FE scheme here (remove pass)
            p_new[i] = p[i] - 1*(c*dt/dx)*(p[i] - p[i - 1]) 
    return p_new
    
def run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt,
                      central=True, square=True):
    """Run sumulation by evaluating each time step."""

    x, p_all = initialize_1D(p0, L, Nx, T, Nt, square=square)
    
    for t in range(Nt):
        p = advection_1D(p_all[t], dx, dt, c, Nx, central=central)
        p_all[t + 1] = p
        
    return x, p_all
    
def plot_1D(x, p, step=0):
    """Plot phi(x, t) at a given time step."""
    fig = plt.figure()
    ax = plt.axes(xlim=(0, round(x.max())),
                  ylim=(0, int(np.ceil(p[0].max())) + 1))  
    ax.plot(x, p[step], marker='.')
    plt.xlabel('$x$ [m]')
    plt.ylabel('Amplitude, $phi$ [$-$]')
    plt.title('Advection in 1D')
    plt.show()
    
def plot_1D_all():
    """Create animation of phi(x, t) for all t."""
    check_variables_1D()
    
    play = widgets.Play(min=0, max=Nt-1, step=1, value=0,
                        interval=100, disabled=False)
    slider = widgets.IntSlider(min=0, max=Nt-1, step=1, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    
    interact(plot_1D, x=fixed(x), p=fixed(p_all), step=play)

    return widgets.HBox([slider])
    
def check_variables_1D():
    """Print current variable values.
    
    Students define CFL.
    """
    print('Current variables values:')
    print(f'  p0 [---]: {p0:0.2f}')
    print(f'  c  [m/s]: {c:0.2f}')
    print(f'  L  [ m ]: {L:0.1f}')
    print(f'  Nx [---]: {Nx:0.1f}')
    print(f'  T  [ s ]: {T:0.1f}')
    print(f'  Nt [---]: {Nt:0.1f}')
    print(f'  dx [ m ]: {dx:0.2e}')
    print(f'  dt [ s ]: {dt:0.2e}')
    print(f'Using central difference?: {central}')
    print(f'Using square init. cond.?: {square}')
    calculated_CFL = c*dt/dx
    if calculated_CFL is None:
        print('CFL not calculated yet.')
    else:
        print(f'CFL: {calculated_CFL:.2e}')
```

<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <p>
        <b>Start of code to define a dictionary to help keep track of analysis "cases." This is not part of the handout.</b>
    </p>
</div>


To help present specific cases in the solution, a dictionary is used to store the key Python variable values that define the problem of interest are stored:
```
p0, c, L, Nx, T, Nt, dx, dt, central
```
Each case is stored as an elemetn of a list `C`, and a few functions are created to facilitate usage.

```python
def case_create(p0=2.0, c=5.0, L=2.0, Nx=100,
                T=4.0, Nt=1000,
                central=True, square=True):
    '''Store variables defining case in a dict.
    
    For use in the solution.
    '''
    C = {}
    C['p0'] = p0
    C['c'] = c
    C['L'] = L
    C['Nx'] = Nx
    C['T'] = T
    C['Nt'] = Nt
    
    C['dx'] = L/Nx
    C['dt'] = T/Nt
    
    C['central'] = central
    C['square'] = square
    
    return C

def case_set(C):
    return (C['p0'], C['c'], C['L'], C['Nx'],
            C['T'], C['Nt'], C['dx'], C['dt'],
            C['central'], C['square'])
```

Define a few cases of interest:

```python
C = []
C.append(case_create())
C[0]['name'] = 'Case 0: default values, central diff, square'

C.append(case_create(square=False))
C[-1]['name'] = 'Case 1: default values, central diff, smooth'

C.append(case_create(central=False))
C[-1]['name'] = 'Case 2: default values, backward diff, square'

C.append(case_create(central=False))
C[-1]['name'] = 'Case 3: default values, backward diff, smooth'

C.append(case_create(L=10, Nx=500, Nt=10000))
C[-1]['name'] = 'Case 4: this is unstable, more slowly, central diff'

C.append(case_create(L=10, Nx=100, T=4, Nt=10000, central=False))
C[-1]['name'] = 'Case 5: this will show numerical diffusion, backward diff'
```

Choose a case and check variable values:

```python
p0, c, L, Nx, T, Nt, dx, dt, central, square = case_set(C[3])
check_variables_1D()
```

<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <p>
        <b>End of dictionary code.</b> Note that it is used below, for example with <code>case_set</code>
    </p>
</div>


Variables are set below, then you should use the functions provided, for example, `check_variables_1D`, prior to running a simulation to make sure you are solving the problem you think you are!

```python
p0 = 2.0
c = 5.0

L = 2.0
Nx = 100
T = 40
Nt =  100000

dx = L/Nx
dt = T/Nt

central = True
square = True
```

_You can ignore any warning that result from running the code below._

```python
check_variables_1D()
x, p_all = run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt, central, square)
```

Use the plotting function to check your initial values. It should look like a "box" shape somwhere in the $x$ domain with velocity $c$ m/s.

```python
plot_1D(x, p_all, step=1500)
```

Visualize. At the very beginning, you should see the wave moving from the left to right. What happens afterwards?

```python
plot_1D_all()
```

## Part 2: Central Difference issues!

The discretization is unstable (regardless of the time step used), largely due to weighting equally the influence by adjacent volumes in the fluxes. The hyperbolic nature of the equation implies that more weight should be given to the upstream/upwind $\phi$ values.

You might think that the initial abrupt edges of the square wave is responsible for the instability. You can test this by replacing the square pulse with a smooth one.



<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2:</b>
Run the 1D simulation again using a smooth pulse. You can do this by changing the value of `square` from `True` to `False`. Does the simulation work?
</div>

```python
square=False
x, p_all = run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt, central, square)
plot_1D_all()
```

## Part 3: Upwind scheme

More weight can be given to the upwind cells by choosing $\phi$ values for the East face $\phi_i$, and for the West face, use $\phi_{i-1}$. This holds true for positive flow directions. For negative flow directions, you should choose $\phi$ values for the East face $\phi_{i-1}$, and for the West face, use $\phi_{i}$. Note that this is less accurate than the central diffence approach but it will ensure stability.



<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.1:</b>

Derive the upwind scheme and apply Forward Euler to the resulting ODE. Then implement it in the function <code>advection_1D</code>. Re-run the analysis after setting the <code>central</code> flag to <code>False</code>.
    
</div>

```python
square=True
central=False

x, p_all = run_simulation_1D(p0, c, L, Nx, T, 10000, dx, dt, central, square)
plot_1D_all()
```


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.2:</b>

In the previous task you should have seen that the method is unstable. Experiment with different time step $\Delta t$ to see if you can find a limit above/below which the method is stable. Write down your $\Delta t$ values and whether or not they were stable, as we will reflect on them later.
    
</div>

```python
# you can change variable values and rerun the analysis here.
```

| $\Delta t$ | stable or unstable? |
| :---: | :---: |
|  |  |

**SOLUTION NEEDED**


## Part 4: False Diffusion

In the previous tasks, we saw how upwinding can be an effective way to handle the type of PDE we are studying (in this case hyperbolic). Now, we will consider _another_ aspect of stability: that of the time integration scheme.

Let’s play with the code. In convective kinematics a von Neumann analysis on the advection equation suggests that the following must hold for stability:
$$
CFL = \frac{c \Delta t}{\Delta x} \leq 1
$$
    
$CFL$ is the Courant–Friedrichs–Lewy condtion, a dimensionless quantity that relates the speed of information leaves a finite volume, relating speed to the ratio of time step duration and cell length. The ratio can provide us an indication of the inherent stability of explicit schemes. 




<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.1:</b>
If you have not already done so, modify the function above to calculate the CFL and run the function <code>check_variables_1D()</code> to check the values. Evaluate the CFL for the time steps you tried in Task 3.2 and write them below, along with the statement of whether or not the scheme was stable.
</div>


_Write your CFL values here, along with the result:_

| CFL | $\Delta t$ | stable or unstable? |
| :---: | :---: | :---: |
|  |  |  |

```python
choose_case = 5
p0, c, L, Nx, T, Nt, dx, dt, central, square = case_set(C[choose_case])
print(C[choose_case]['name'])
```

```python
x, p_all = run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt, central, square)
plot_1D_all()
```


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.2:</b>

Now use the CFL to compute the time step $\Delta t$ that defines the boundary between the stable and unstable region. Then re-run the analysis for this value, as well as a value that is slightly above and below that threshold $\Delta t$.
</div>

```python
# SOLUTION
```

### So you think everything is stable and perfect, right?

Based on the previous task, it looks like we have a good handle on this problem, and are able to use the CFL to set up a reliable numerical scheme for all sorts of complex problems---right?!

**WRONG!**

Remember, in this problem we are dealing with single "wave" propagating at a _constant_ speed. In practice we apply numerical schemes to more complex methods. For example, most problems consider _variable_ speed/velocity in more than one dimension. When this is the case, the problem cannot be described by a single CFL value! As a rule of thumb, a modeller would then choose a conservative CFL value, determined by the largest expected flow velocities (in the case of a regular mesh).

Let's apply this concept by using a CFL condition of 0.8 to visualize the impact on the numerical solution. 



<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.3:</b>

Find $\Delta t$ such that CFL is 0.8 and re-run the analysis. What do you observe?

<em>Make sure you look at the complete solution, not just the first few steps.</em>
    
</div>

```python
# SOLUTION
```


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.4:</b>

Describe what you observe in the result of the previous task and state (yes/no) whether or not this should be expected, given the PDE we are solving. Explain your answer in a couple sentences.
    
</div>

<!-- #region -->
<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution:</b>

The phenomenon you saw in the previous task is <em>numerical</em>, or <em>false</em> diffusion.
...


No!

- we observe the wave diffuses
- our PDE only considers convection
- where does this come from?!


</p>
</div>
<!-- #endregion -->

## Part 5: 2D Implementation in Python


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 5.1:</b>
Apply FVM by hand to the 2D advection equation. The volumes are rectangular. This is a good example of an exam problem.

$$
\frac{\partial \phi}{\partial t} + c_x\frac{\partial \phi}{\partial x} + c_y\frac{\partial \phi}{\partial y} = 0
$$

</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 5.2:</b>
The code is set up in a very similar way to the 1D case above. Use it to explore how the advection problem works in 2D! In particular, see if you observe the effect called "numerical diffusion" --- when the numerical scheme causes the square pulse to "diffuse" into a bell shaped surface. Even though only the advection term was implmented!
</div>


<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <p>
        The initial values of the variables below will result in numerical instability. See if you can fix it!
    </p>
</div>

```python
p0 = 2.0
cx = 5.0
cy = 5.0

Lx = 2.0
Nx = 100
Ly = 2.0
Ny = 100
T = 40
Nt =  900

dx = Lx/Nx
dy = Ly/Ny
dt = T/Nt

central = True

```

```python
def initialize_2D(p0, Lx, Nx, Ly, Ny, T, Nt):
    x = np.linspace(dx/2, Lx - dx/2, Nx)
    y = np.linspace(dy/2, Ly - dx/2, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialise domain: cubic pulse with p0 between 0.5 and 1
    p_init = np.zeros((Nx, Ny))
    p_init[int(0.5/dx):int(1/dx + 1), int(0.5/dy):int(1/dy + 1)] = p0

    p_all = np.zeros((Nt + 1, Nx, Ny))
    p_all[0] = p_init
    return X, Y, p_all

def advection_2D(p, cx, cy, Nx, Ny, dx, dy, dt, central=True):

    p_new = np.ones((Nx,Ny))

    for i in range(0, Nx):
        for j in range(0, Ny):
            if central:
                p_new[i-1,j-1] = (p[i-1,j-1]
                                  - 0.5*(cx*dt/dx)*(p[i,j-1] - p[i-2,j-1])
                                  - 0.5*(cy*dt/dy)*(p[i-1,j] - p[i-1,j-2]))
            else:
                p_new[i, j] = (p[i, j] - (cx*dt/dx)*(p[i, j] - p[i - 1, j]) 
                                       - (cy*dt/dy)*(p[i, j] - p[i, j - 1]))
    return p_new
    
def run_simulation_2D(p0, cx, cy, Lx, Nx, Ly, Ny,
                      T, Nt, dx, dy, dt, central=True):
    
    X, Y, p_all = initialize_2D(p0, Lx, Nx, Ly, Ny, T, Nt)
    
    for t in range(Nt):
        p = advection_2D(p_all[t], cx, cy, Nx, Ny,
                         dx, dy, dt, central=central)
        p_all[t + 1] = p
        
    return X, Y, p_all

def plot_2D(p, X, Y, step=0):
    'Create 2D plot, X and Y are formatted as meshgrid.'''
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('$\phi$ [-]') 
    ax.set_title('Advection in 2D')
    surf = ax.plot_surface(X, Y, p[step],
                           cmap='Blues', rstride=2, cstride=2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_2D_all():
    check_variables_2D()
    
    play = widgets.Play(min=0, max=Nt-1, step=1, value=0,
                        interval=100, disabled=False)
    slider = widgets.IntSlider(min=0, max=Nt-1,
                               step=1, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    
    interact(plot_2D,
             p=fixed(p_all),
             X=fixed(X),
             Y=fixed(Y),
             step=play)

    return widgets.HBox([slider])

def check_variables_2D():
    print('Current variables values:')
    print(f'  p0 [---]: {p0:0.2f}')
    print(f'  cx [m/s]: {cx:0.2f}')
    print(f'  cy [m/s]: {cy:0.2f}')
    print(f'  Lx [ m ]: {Lx:0.1f}')
    print(f'  Nx [---]: {Nx:0.1f}')
    print(f'  Ly [ m ]: {Ly:0.1f}')
    print(f'  Ny [---]: {Ny:0.1f}')
    print(f'  T  [ s ]: {T:0.1f}')
    print(f'  Nt [---]: {Nt:0.1f}')
    print(f'  dx [ m ]: {dx:0.2e}')
    print(f'  dy [ m ]: {dy:0.2e}')
    print(f'  dt [ s ]: {dt:0.2e}')
    print(f'Using central difference?: {central}')
    print(f'Solution shape p_all[t_i]: ({Nx}, {Ny})')
    print(f'Total time steps in p_all: {Nt+1}')
    print(f'CFL, direction x: {cx*dt/dx:.2e}')
    print(f'CFL, direction y: {cy*dt/dy:.2e}')
```

<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
Here you can check the Python variable assignments and visualize the initial conditions in the plot. Notice that we already redefine the time parameters to give a CFL of less than 1, in both directions.
</p>
</div>

```python
T = 1
Nt =  1000
dt = T/Nt
check_variables_2D()
X, Y, p_all = initialize_2D(p0, Lx, Nx, Ly, Ny, T, Nt)
plot_2D(p_all, X, Y)
```

<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
By default the central difference scheme is used, so you should see the numerical instability just as we did before in the 1D case.

</p>
</div>

```python
X, Y, p_all = run_simulation_2D(p0, cx, cy, Lx, Nx, Ly, Ny, T, Nt, dx, dy, dt, central)
plot_2D_all()
```

<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
Switching to the upwind scheme gives a stable solution. Note the numerical diffusion (change from cubic to bell shape). Also, as the edges of the bell shape spread outside of the X-Y domain, you will see them appear on the corners: this is a result of the fact that the iteration loops over X and Y in the meshgrid format don't properly find the "neighbor" volumes. To understand this, refer to the textbook where the matrix form of the 2D solution is presented and see how the index of the neighbor volumes is not sequential as in the for loops.

</p>
</div>

```python
X, Y, p_all = run_simulation_2D(p0, cx, cy, Lx, Nx, Ly, Ny, T, Nt, dx, dy, dt, central=False)
plot_2D_all()
```

<!-- #region -->
**End of notebook.**

<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
  <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
  </div>
  <div style="font-size: 75%; margin-top: 10px; text-align: right;">
    &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. 
    This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
