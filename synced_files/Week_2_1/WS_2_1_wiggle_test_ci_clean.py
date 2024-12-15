import numpy as np
import matplotlib.pylab as plt
from ipywidgets import interact, fixed, widgets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            pass # add central averaging + FE scheme here (remove pass)
        else:
            pass # add upwind + FE scheme here (remove pass)
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
    calculated_CFL = None
    if calculated_CFL is None:
        print('CFL not calculated yet.')
    else:
        print(f'CFL: {calculated_CFL:.2e}')

p0 = 2.0
c = 5.0

L = 2.0
Nx = 100
T = 40
Nt =  10000

dx = L/Nx
dt = T/Nt

central = True
square = True

check_variables_1D()
x, p_all = run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt, central, square)

plot_1D(x, p_all)

plot_1D_all()

square=False
x, p_all = run_simulation_1D(p0, c, L, Nx, T, Nt, dx, dt, central, square)
plot_1D_all()

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

T = 1
Nt =  1000
dt = T/Nt
check_variables_2D()
X, Y, p_all = initialize_2D(p0, Lx, Nx, Ly, Ny, T, Nt)
plot_2D(p_all, X, Y)

X, Y, p_all = run_simulation_2D(p0, cx, cy, Lx, Nx, Ly, Ny, T, Nt, dx, dy, dt, central)
plot_2D_all()

X, Y, p_all = run_simulation_2D(p0, cx, cy, Lx, Nx, Ly, Ny, T, Nt, dx, dy, dt, central=False)
plot_2D_all()

