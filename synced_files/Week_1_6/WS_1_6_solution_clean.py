
import numpy as np
import matplotlib.pyplot as plt

dt =5                          
t_end = 60  
Ts =  20       # [C] 
k = 0.5        # [s^-1]

t = np.arange(0,t_end+dt,dt)
n = len(t)
T = np.empty(n)
T[0] = 50

for i in range(n-1):
    T[i+1] = T[i]-k*(T[i]-Ts)*dt 
    
plt.plot(t, T, 'o-')
plt.xlabel('t (s)')
plt.ylabel('T (deg)')
plt.grid()

dt = 10                          
t_end = 60  
Ts =  20       # [C] 
k = 0.5        # [s^-1]

t = np.arange(0,t_end+dt,dt)
n = len(t)
T = np.empty(n)
T[0] = 50     

for i in range(n-1):
    T[i+1] = (T[i]+k*Ts*dt)/(1+k*dt) 
    
plt.plot(t, T, 'o-')
plt.xlabel('t (s)')
plt.ylabel('T (deg)')
plt.grid()

import numpy as np 
import matplotlib.pyplot as plt

Ts = 30
alpha = 500
dx=0.02

x = np.arange(0,0.1+dx,dx)
T = np.zeros(x.shape)
n=len(x)

T[0] = 250
T[-1] = Ts

matrix_element = -(2+dx**2*alpha)
A = np.zeros((len(x)-2,len(x)-2))
np.fill_diagonal(A, matrix_element)
A[np.arange(n-3), np.arange(1, n-2)] = 1  # Upper diagonal
A[np.arange(1, n-2), np.arange(n-3)] = 1  # Lower diagonal

b_element = -dx**2*alpha*Ts
b = np.zeros(len(x)-2) + b_element
b[0] = b[0] - T[0]
b[-1] = b[-1] - T[-1]

T[1:-1] = np.linalg.solve(A,b)

plt.plot(x,T,'*',label='Estimated solution')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Estimated solution using Central Difference method')
plt.legend()
plt.show()

print(f'The estimated temperature at the nodes are: {[f"{temp:.2f}" for temp in T]} [C]')

import numpy as np
import matplotlib.pyplot as plt
 
Ts = 30
alpha = 500
dx=0.02
 
x = np.arange(0,0.1+dx,dx)
T = np.zeros(x.shape)
n=len(x)
 
T[0] = 250
 
 
matrix_element = -(2+dx**2*alpha)
A = np.zeros((len(x)-2,len(x)-2))
np.fill_diagonal(A, matrix_element)
A[np.arange(n-3), np.arange(1, n-2)] = 1  # Upper diagonal
A[np.arange(1, n-2), np.arange(n-3)] = 1  # Lower diagonal
 
A[-1,-1] = -(1+dx**2*alpha)  # the lower right corner of the matrix changes
 
b_element = -dx**2*alpha*Ts
b = np.zeros(len(x)-2) + b_element
b[0] = b[0] - T[0]
b[-1] = b[-1]               # the vector b also changes
 

T[1:-1] = np.linalg.solve(A,b)
T[-1] = T[-2]  # this line has been added
 
plt.plot(x,T,'*',label='Estimated solution')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Estimated solution using Central Difference method with Neumann BC')
plt.legend()
plt.show()
 
print(f'The estimated temperature at the nodes are: {[f"{temp:.2f}" for temp in T]} [C]')

import numpy as np 
import matplotlib.pyplot as plt

Ts = 30
alpha = 500
dx=0.0004 # The grid size is reduced to (0.02^2 = 0.0004)

x = np.arange(0,0.1+dx,dx)
T = np.zeros(x.shape)
n=len(x)

T[0] = 250
T[-1] = Ts

matrix_element = 1-alpha*dx**2    # The matrix element changes (from CD)
A = np.zeros((len(x)-2,len(x)-2))
np.fill_diagonal(A, -2)            # different from CD     
A[np.arange(n-3), np.arange(1, n-2)] = 1  # Upper diagonal
A[np.arange(1, n-2), np.arange(n-3)] = matrix_element  # Lower d.: different from CD
print(A)

b_element = -dx**2*alpha*Ts
b = np.zeros(len(x)-2) + b_element
b[0] = b[0] - matrix_element * T[0]
b[-1] = b[-1] - T[-1]

T[1:-1] = np.linalg.solve(A,b)

plt.plot(x,T,'*',label='Estimated solution')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Estimated solution using Forward Difference method')
plt.legend()
plt.show()

print(f'The x values =============================: {[f"{x_loc:.4f}" for x_loc in x]} [m]')
print(f'The estimated temperature at the nodes are: {[f"{temp:.2f}" for temp in T]} [C]')

ind = np.argmin(abs(x-0.02))
print(f'The temperature at x=0.02 is: {T[ind]:.2f} [C]')

def gauss_jordan(A, b):
    """
    Solves the system of linear equations Ax = b using Gauss-Jordan elimination.
    
    Parameters:
    A (numpy.ndarray): Coefficient matrix (n x n).
    b (numpy.ndarray): Right-hand side vector (n).
    
    Returns:
    numpy.ndarray: Solution vector (x) if the system has a unique solution.
    """
    # Form the augmented matrix [A | b]
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    aug_matrix = np.hstack([A, b.reshape(-1, 1)])
    
    n = len(b)  # Number of rows (or variables)
    
    for i in range(n):
        # Partial pivoting to handle zero diagonal elements (optional, but more robust)
        max_row = np.argmax(np.abs(aug_matrix[i:, i])) + i
        if aug_matrix[max_row, i] == 0:
            raise ValueError("The matrix is singular and cannot be solved.")
        if max_row != i:
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
        
        # Make the diagonal element 1
        aug_matrix[i] = aug_matrix[i] / aug_matrix[i, i]
        
        # Make all other elements in the current column 0
        for j in range(n):
            if j != i:
                aug_matrix[j] -= aug_matrix[j, i] * aug_matrix[i]
    
    # Extract the solution (last column of the augmented matrix)
    return aug_matrix[:, -1]

import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

start_time = time.time()
A_inv = np.linalg.inv(A)
T[1:-1] = A_inv @ b
time_used_0 = time.time() - start_time
print(f"The time used by direct matrix inversion solution is {time_used_0: 0.3e} sec")
assert np.allclose(np.dot(A, T[1:-1]), b), "Oops! The calculation is wrong.."

start_time = time.time()
u1 = gauss_jordan(A, b)
time_used_1 = time.time() - start_time
print(f"The time used by Gauss-jordan solution is {time_used_1: 0.3e} sec")
assert np.allclose(np.dot(A, u1), b), "Oops! The calculation is wrong.."

start_time = time.time()
A_sparse = csc_matrix(A)# Convert A to a compressed sparse column (CSC) matrix
u2 = spsolve(A_sparse, b)
time_used_2 = time.time() - start_time
print(f"The time used by the sparse matrix solver is {time_used_2: 0.3e} sec")
assert np.allclose(np.dot(A, u2), b), "Oops! The calculation is wrong.."

