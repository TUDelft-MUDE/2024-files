# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dt =YOUR_CODE_HERE                          
t_end =YOUR_CODE_HERE  
Ts =  20       # [C] 
k = 0.5        # [s^-1]

t = YOUR_CODE_HERE
n = YOUR_CODE_HERE
T = YOUR_CODE_HERE
T[0] = YOUR_CODE_HERE



# ----------------------------------------
for i in range(n-1):
    T[i+1] = YOUR_CODE_HERE
    
plt.plot(t, T, 'o-')
plt.xlabel('t (s)')
plt.ylabel('T (deg)')
plt.grid()

# ----------------------------------------
dt = YOUR_CODE_HERE                          
t_end = 60  
Ts =  20       # [C] 
k = 0.5        # [s^-1]

t = YOUR_CODE_HERE
n = YOUR_CODE_HERE
T = YOUR_CODE_HERE
T[0] = YOUR_CODE_HERE

for i in range(n-1):
    T[i+1] = YOUR_CODE_HERE 
    
plt.plot(t, T, 'o-')
plt.xlabel('t (s)')
plt.ylabel('T (deg)')
plt.grid()

# ----------------------------------------
import numpy as np 
import matplotlib.pyplot as plt

Ts = 30
alpha = 500
dx=0.02

# grid creation
x = np.arange(0,0.1+dx,dx)
T = np.zeros(x.shape)
n=len(x)

# boundary conditions
T[0] = 250
T[-1] = Ts

# Building matrix A
matrix_element = -(2+dx**2*alpha)
A = np.zeros((len(x)-2,len(x)-2))
np.fill_diagonal(A, matrix_element)
A[np.arange(n-3), np.arange(1, n-2)] = 1  # Upper diagonal
A[np.arange(1, n-2), np.arange(n-3)] = 1  # Lower diagonal
print(A.shape)
# Building vector b
b_element = -dx**2*alpha*Ts
b = np.zeros(len(x)-2) + b_element
b[0] = b[0] - T[0]
b[-1] = b[-1] - T[-1]

# Solving the system
T[1:-1] = np.linalg.solve(A,b)

plt.plot(x,T,'*',label='Estimated solution')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Estimated solution using Central Difference method')
plt.legend()
plt.show()

print(f'The estimated temperature at the nodes are: {[f"{temp:.2f}" for temp in T]} [C]')

# ----------------------------------------
YOUR_CODE_HERE

# ----------------------------------------
Ts = 30
alpha = 500
dx=0.02
 
# grid creation
x = np.arange(0,0.1+dx,dx)
T = np.zeros(x.shape)
n=len(x)
 
# boundary conditions
T[0] = 250
 
 
# Building matrix A
matrix_element = -(2+dx**2*alpha)
A = np.zeros((len(x)-2,len(x)-2))
np.fill_diagonal(A, matrix_element)
A[np.arange(n-3), np.arange(1, n-2)] = 1  # Upper diagonal
A[np.arange(1, n-2), np.arange(n-3)] = 1  # Lower diagonal
print(A.shape)
A[-1,-1] = -(1+dx**2*alpha)  #the matrix changes
 
# Building vector b
b_element = -dx**2*alpha*Ts
b = np.zeros(len(x)-2) + b_element
b[0] = b[0] - T[0]
b[-1] = b[-1]               #the vector b also changes
 
# Solving the system

T[1:-1] = np.linalg.solve(A,b)
T[-1] = T[-2]  

plt.plot(x,T,'*',label='Estimated solution')
plt.xlabel('x')
plt.ylabel('T')
plt.title('Estimated solution using Central Difference method')
plt.legend()
plt.show()
 
print(f'The estimated temperature at the nodes are: {[f"{temp:.2f}" for temp in T]} [C]')

# ----------------------------------------
YOUR_CODE_HERE

# ----------------------------------------
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

# ----------------------------------------
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Inverted matrix solution
start_time = time.time()
A_inv = np.linalg.inv(A)
T[1:-1] = A_inv @ b
time_used_0 = time.time() - start_time
print(f"The time used by direct matrix inversion solution is {time_used_0: 0.3e} sec")
assert np.allclose(np.dot(A, T[1:-1]), b), "Oops! The calculation is wrong.."


# Gauss-jordan solution
start_time = time.time()
u1 = gauss_jordan(A, b)
time_used_1 = time.time() - start_time
print(f"The time used by Gauss-jordan solution is {time_used_1: 0.3e} sec")
#Check if the solution is correct:
assert np.allclose(np.dot(A, u1), b), "Oops! The calculation is wrong.."

# Solution by a sparse matrix solver 
start_time = time.time()
A_sparse = csc_matrix(A)# Convert A to a compressed sparse column (CSC) matrix
u2 = spsolve(A_sparse, b)
time_used_2 = time.time() - start_time
print(f"The time used by the sparse matrix solver is {time_used_2: 0.3e} sec")
#Check if the solution is correct:
assert np.allclose(np.dot(A, u2), b), "Oops! The calculation is wrong.."

