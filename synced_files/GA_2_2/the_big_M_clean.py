# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import matplotlib.pyplot as plt
import numpy as np

def readGmsh (fname):
    if not fname.endswith('.msh'):
        raise RuntimeError('Unexpected mesh file extension')
    nodes = []
    elems = []
    parse_nodes = False
    parse_elems = False
    rank = 2
    nnodes = 3
    with open(fname) as msh:
        for line in msh:
            sp = line.split()
            if '$Nodes' in line:
                parse_nodes = True
            elif '$Elements' in line:
                parse_nodes = False
                parse_elems = True
            elif parse_nodes and len(sp) > 1:
                if len(sp[1:]) != 3:
                    raise SyntaxError('readGmsh: Three coordinates per node are expected')
                coords = np.array(sp[1:], dtype=np.float64)
                nodes.append(coords)
            elif parse_elems and len(sp) > 1:
                eltype = int(sp[1])
                inodes = np.array(sp[3 + int(sp[2]):], dtype=np.int32) - 1
                elems.append(inodes)
    return np.array(nodes), np.array(elems)

def plotMesh(nodes, connectivity):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.triplot(nodes[:,0], nodes[:,1], connectivity, 'k-', lw=0.2)
    plt.axis('equal')
    plt.axis('off')

nodes, connectivity = readGmsh('bigM.msh')
plotMesh(nodes, connectivity)

# %% [markdown]

# %%
from scipy import sparse
from scipy.sparse import linalg

def get_shape_functions_T3( n1, n2, n3 ):
    coordinate_Matrix = np.ones((3,3))
    coeffs = np.zeros((3,3))
    
    coordinate_Matrix[0, 1] = n1[0]
    coordinate_Matrix[0, 2] = n1[1]
    coordinate_Matrix[1, 1] = n2[0]
    coordinate_Matrix[1, 2] = n2[1]
    coordinate_Matrix[2, 1] = n3[0]
    coordinate_Matrix[2, 2] = n3[1]
    
    
    c_inv = np.linalg.inv(coordinate_Matrix)
    coeffs[0] = np.dot(c_inv, (1,0,0))
    coeffs[1] = np.dot(c_inv, (0,1,0))
    coeffs[2] = np.dot(c_inv, (0,0,1))
    
    return coeffs

def get_B_matrix(n1, n2, n3):
    coeffs = get_shape_functions_T3(n1, n2, n3)
    B_matrix = [[coeffs[0][1], coeffs[1][1], coeffs[2][1]], [coeffs[0][2], coeffs[1][2], coeffs[2][2]]]
    
    return np.array(B_matrix)

def get_area(n1, n2, n3):
    
    u = n3[0:2]-n1[0:2]
    v = n2[0:2]-n1[0:2]
  
    return np.abs(np.cross(u, v))/2

def get_element_K(n1, n2, n3, nu):
    B = get_B_matrix(n1, n2, n3)
    element_area = get_area(n1, n2, n3)
    return element_area*np.dot(np.transpose(B), B)*nu

def get_global_K(nodes, connectivity, n_elem, nu):
    n_DOF = len(nodes)
    K = np.zeros((n_DOF, n_DOF))
    
    for i in range(n_elem):
        elnodes = connectivity[i,:]
        K_el = get_element_K(nodes[elnodes[0]], nodes[elnodes[1]], nodes[elnodes[2]], nu)
        K[np.ix_(elnodes,elnodes)] += K_el
                
    return K

def get_global_f(nodes, connectivity, q):
    n_DOF = len(nodes)
    f = np.zeros(n_DOF)
    
    for elem in connectivity:
        area = get_area(nodes[elem[0]], nodes[elem[1]], nodes[elem[2]])
        for i in range(3):
            f[elem[i]] += q*area/3
            
    return f

def evaluate_N_matrix(ipcoords, n1, n2, n3):    
    coeffs = get_shape_functions_T3(n1, n2, n3)
    
    x = ipcoords[0]
    y = ipcoords[1]
    
    N_matrix = [[coeffs[0][0] + coeffs[0][1]*x + coeffs[0][2]*y, coeffs[1][0] + coeffs[1][1]*x + coeffs[1][2]*y, coeffs[2][0] + coeffs[2][1]*x + coeffs[2][2]*y]]
    
    return N_matrix

def get_element_M(n1, n2, n3):
    M_el = np.zeros((3,3))
    element_area = get_area(n1, n2, n3)    
    
    
    integration_locations = [(n1+n2)/2,  (n2+n3)/2,  (n3+n1)/2]
    integration_weights = [element_area/3, element_area/3, element_area/3]
    
    
    for x_ip, w_ip in zip(integration_locations, integration_weights):
        N_local = evaluate_N_matrix(x_ip, n1, n2, n3)
        M_el += np.dot(np.transpose(N_local), N_local)*w_ip
    return M_el
    

def get_global_M(nodes, connectivity, n_elem):
    n_DOF = len(nodes)
    M = np.zeros((n_DOF, n_DOF))
    
    for i in range(n_elem):
        elnodes = connectivity[i,:]
        M_el = get_element_M(nodes[elnodes[0]], nodes[elnodes[1]], nodes[elnodes[2]])
        M[np.ix_(elnodes,elnodes)] += M_el

    return M

# %% [markdown]

# %%

dt = 0.005
nt = 500
nu = 1
q = 15
T_initial = 30
T_edge = 10

constrained_nodes = []
free_nodes = []
n_free = 0
min_y = min(nodes[:,1])
min_x = min(nodes[:,0])
max_x = max(nodes[:,0])
for i in range(len(nodes)):
    if nodes[i,1] == min_y and nodes[i,0] > 0.5*max_x:
            constrained_nodes.append(i)
    else:
        n_free += 1
        free_nodes.append(i) 

n_DOF = len(nodes)
us = np.zeros((nt+1,n_DOF))

us[0] = T_initial

f = get_global_f(nodes, connectivity, q)
Kdense = get_global_K(nodes, connectivity, len(connectivity), nu)
Mdense = get_global_M(nodes, connectivity, len(connectivity))

K = sparse.csc_matrix(Kdense)
M = sparse.csc_matrix(Mdense)

Kmod = K + M/dt
Kmodff = Kmod[free_nodes,:][:,free_nodes]
Kmodfp = Kmod[free_nodes,:][:,constrained_nodes]

solver = linalg.factorized(Kmodff)

for i in range(nt):
    us[i+1, constrained_nodes] = T_edge
    fmod = M.dot(us[i]) / dt + f
    ff = fmod[free_nodes] - Kmodfp.dot(us[i+1, constrained_nodes])
    us[i+1, free_nodes] = solver(ff)

# %% [markdown]

# %% [markdown]

# %%

from ipywidgets import interact, fixed, widgets
from matplotlib import colors
from matplotlib import cm

def plot_result(nodes, result, x_lim, y_lim):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    
    norm = colors.Normalize(vmin=8, vmax=52)
    
    
    x = nodes[:,0]
    y = nodes[:,1]
    ax.plot_trisurf(x, y, result, triangles=connectivity, norm=norm, cmap = cm.coolwarm)
    
    
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    ax.set_zlim((0, 55))
    
    
    cmap = cm.ScalarMappable(norm = colors.Normalize(8, 52), cmap = cm.coolwarm)
    cmap.set_array(result)
    fig.colorbar(cmap, ax=ax)

time_step = 5

plot_result(nodes, us[time_step], (min_x, max_x), (min_x,max_x))

# %%

from ipywidgets import interact, fixed, widgets
from matplotlib import colors
from matplotlib import cm

def plot_result3d(nodes, conn, results, step):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    
    norm = colors.Normalize(vmin=8, vmax=52)
    
    
    x = nodes[:,0]
    y = nodes[:,1]
    
    ax.plot_trisurf(x, y, results[step], triangles=conn, norm=norm, cmap = cm.coolwarm)
    
    
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    ax.set_zlim((0, 55))
    
    
    cmap = cm.ScalarMappable(norm = colors.Normalize(8, 52), cmap = cm.coolwarm)
    cmap.set_array(results[step])
    fig.colorbar(cmap, ax=ax)
    
play = widgets.Play(min=0, max=nt-1, step=1, value=0, interval=100, disabled=False)
slider = widgets.IntSlider(min=0, max=nt-1, step=1, value=0)
widgets.jslink((play, 'value'), (slider, 'value'))
interact(plot_result3d,
         nodes = fixed(nodes),
         conn = fixed(connectivity),
         results = fixed(us),
         step = play)
         
widgets.HBox([play, slider])

# %%

def plot_result(nodes, conn, results, step):
    fig = plt.figure()
    ax = fig.add_subplot()
    x = nodes[:,0]
    y = nodes[:,1];
    cscale = colors.Normalize(8, 52)
    ax.set_aspect('equal')
    tcf = ax.tricontourf(x, y, conn, results[step], norm=cscale, cmap=cm.coolwarm, levels=12)
    ax.triplot(x,y,conn, 'k-', lw=0.2)
    
    cmap = cm.ScalarMappable(norm = cscale, cmap = cm.coolwarm)
    cmap.set_array(results[step])
    fig.colorbar(cmap, ax=ax)
    plt.axis('off')

play = widgets.Play(min=0, max=nt-1, step=1, value=0, interval=100, disabled=False)
slider = widgets.IntSlider(min=0, max=nt-1, step=1, value=0)
widgets.jslink((play, 'value'), (slider, 'value'))
interact(plot_result,
         nodes = fixed(nodes),
         conn = fixed(connectivity),
         results = fixed(us),
         step = play)
         
widgets.HBox([play, slider])

# %% [markdown]

