# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def evaluate_N(x_local, dx):
    return np.array([[1-x_local/dx, x_local/dx]])

def evaluate_B(x_local, dx):
    return np.array([[-1/dx, 1/dx]])

def get_element_matrix(EA, k, dx):                  
    
    integration_locations = [(dx - dx/(3**0.5))/2, (dx + dx/(3**0.5))/2]
    integration_weights = [dx/2, dx/2]
    n_ip = len(integration_weights)

    n_node = 2
    K_loc = np.zeros((n_node,n_node))

    for x_ip, w_ip in zip(integration_locations, integration_weights):
        B = evaluate_B(x_ip, dx)
        N = evaluate_N(x_ip,dx)                     
        K_loc += EA*np.dot(np.transpose(B), B)*w_ip
        K_loc += k*np.dot(np.transpose(N), N)*w_ip  

    return K_loc

def get_element_force(q, dx):
    n_node = 2
    N = np.zeros((1,n_node))
    
    integration_locations = [(dx - dx/(3**0.5))/2, (dx + dx/(3**0.5))/2]
    integration_weights = [dx/2, dx/2]
    
    f_loc = np.zeros((n_node,1))
    
    for x_ip, w_ip in zip(integration_locations, integration_weights):
        N = evaluate_N(x_ip,dx)
        f_loc += np.transpose(N)*q*w_ip

    return f_loc

def get_nodes_for_element(ie):
    return np.array([ie,ie+1])

def assemble_global_K(rod_length, n_el, k, EA):     
    n_DOF = n_el+1
    dx = rod_length/n_el
    K_global = np.zeros((n_DOF, n_DOF))
    
    for i in range(n_el):
        elnodes = get_nodes_for_element(i)
        K_global[np.ix_(elnodes,elnodes)] += get_element_matrix(EA, k, dx)   
    
    return K_global

def assemble_global_f(rod_length, n_el, q):
    n_DOF = n_el+1
    dx = rod_length/n_el
    f_global = np.zeros((n_DOF,1))
    
    for i in range(n_el):
        elnodes = get_nodes_for_element(i) 
        f_global[elnodes] += get_element_force(q, dx)
        
    return np.squeeze(f_global)

def simulate(n_element,k):                          
    length = 3
    EA = 1e3
    n_node = n_element + 1
    F_right = 10
    u_left = 0 
    q_load = 0                                      

    dx = length/n_element
    x = np.linspace(0,length,n_node)

    K = assemble_global_K(length, n_element, k, EA) 

    f = assemble_global_f(length, n_element, q_load)

    f[n_node-1] += F_right

    u = np.zeros(n_node)

    f -= K[0,:] * u_left
    K_inv = np.linalg.inv(K[1:n_node, 1:n_node])
    u[1:n_node] = np.dot(K_inv, f[1:n_node])
    u[0] = u_left

    return x, u

# %%
import numpy as np
import matplotlib.pyplot as plt

x10, u_with_k = simulate(10,1000)
x10, u_without_k = simulate(10,0)
plt.figure()
plt.plot(x10, u_with_k, label='k=1000')
plt.plot(x10, u_without_k, label='k=0')
plt.xlabel('x')
plt.ylabel('u')
plt.legend();

# %% [markdown]

# %% [markdown]

# %%
x10, u3_10 = simulate(10, 1.e3)
x5, u3_5 = simulate(5, 1.e3)
x2, u3_2 = simulate(2, 1.e3)
plt.plot(x2, u3_2, label='ne=2')
plt.plot(x5, u3_5, label='ne=5')
plt.plot(x10, u3_10, label='ne=10')
plt.xlabel('x')
plt.ylabel('u')
plt.title('k=1000')
plt.legend();

plt.figure()

# %%
x5, u6_5 = simulate(5, 1.e6)
x20, u6_20 = simulate(20, 1.e6)
x100, u6_100 = simulate(100, 1.e6)
plt.plot(x5, u6_5, label='ne=5')
plt.plot(x20, u6_20, label='ne=20')
plt.plot(x100, u6_100, label='ne=100')
plt.xlabel('x')
plt.ylabel('u')
plt.title('k=10^6')
plt.legend();

# %% [markdown]

# %% [markdown]

