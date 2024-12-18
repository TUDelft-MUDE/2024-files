# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"
# # WS 1.6: Understanding Ordinary Differential Equation
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
#
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6. For: 9th October, 2024.*

# %% [markdown]
# ## Overview
#
#
# This assignment is aimed to develop an understanding of the **Ordinary Differential Equation (ODE)**. There will be two sections about cooling and heating scenerios, corresponding to the first-order and the second-order ODEs. Please go through the text that follows and perform all steps outlined therein.
#
# ## Part 1: First-order ODE
#
# In the study of heat transfer, **Newton's law of cooling** is a physical law which states that the rate of heat loss of a body is directly proportional to the difference in the temperatures between the body and its environment. It can be expressed in the form of ODE, as below:
#
# $$\frac{dT}{dt}=-k(T - T_s)$$
#
# where $T$ is the temperature of the object at time $t$, $T_s$ is the temperature of the surrounding and assumed to be constant, and $k$ is the constant that characterizes the ability of the object to exchange the
# heat energy (unit 1/s), which depends on the specific material properties.
#
#
# Now, Let's consider an object with the initial temperature of  50°C in a surrounding environment with constant temperature at 20°C. The constant of heat exchange between the object and the environment is 0.5 $s^{-1}$.
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1:</b>
#  
# Suppose the considered period of time is long enough (bring it to steady state), what will be the final temperature of the object? 
#     
# **Write your answer in the following markdown cell.**
#  
# </p>
# </div>

# %% [markdown]
#

# %% [markdown]
# Next, let's evaluate the temperature of the object by checking it at a series of time points.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.2:</b>
#
# Write the algebraic representation of the ODE using Explicit Euler.
#  
# </p>
# </div>

# %% [markdown]
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3:</b>
#
# Compute the temperature evolution in the next 60 seconds.
#
# **Please complete the missing parts of the code in each step below, which is divided into 5 substeps (a through e).**
#  
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3a:</b> 
#
# The time step of 5 seconds is constant. Discretize the time points, the solution vector $T$ and define the initial condition.
#  
# </p>
# </div>

# %%
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



# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3b:</b> 
#
# Implement your time discretization and find the solution from $t=0$ until $t=60$ sec. 
#  
# </p>
# </div>

# %%
for i in range(n-1):
    T[i+1] = YOUR_CODE_HERE
    
plt.plot(t, T, 'o-')
plt.xlabel('t (s)')
plt.ylabel('T (deg)')
plt.grid()

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3c:</b>
#
# Try different time steps to check the stability of the calculation. At which value the solution is stable?
#  
# </p>
# </div>

# %% [markdown]
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3d:</b>
#
# Obtain the mathematical expression that proves your stability criteria.
#  
# </p>
# </div>

# %% [markdown]
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3e:</b>
#
# Now, discretize the equation using Implicit (Backward) Euler. Can you find a time step that makes the problem unstable?
#  
# </p>
# </div>

# %%
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

# %% [markdown]
# ## Part 2: Second-order ODE
#
# The following 1D equation describes the steady state solution of the temperature along a pin that sticks out of a furnace. The rest of the pin is exposed to the ambient. 
#
# $$
# \frac{d^2T}{dx^2} -\alpha(T-T_s)=0
# $$
#
# The ambient temperature is $T_s= 30^o$ C and the temperature at the wall is $250^o$ C. The length of the pin is 0.1m. Your grid should have a spatial step of 0.02 m. Finally, $\alpha=500$.

# %% [markdown]
#
# The solution includes the steps:
# 1. Use the Taylor series to obtain an approximation for the derivatives;
# 2. Discretize the equation;
# 3. Define parameters and grid;
# 4. Provide boundary conditions;
# 5. Build matrix with solution $AT=b$
# 6. Solve the matrix

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1:</b>
#
# This task has three parts: a) discretize the analytic expression into a system of equations using central differences, b) write the system of equations by hand (e.g., the matrices), and c) implement the discretized system of equations in a code cell.
#
# <em>Parts 2.1b and 2.1c do not need to be completed in order; in fact, it may be useful to go back and forth between the two in order to understand the problem.</em> 
#     
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1a:</b>
#
# Discretize the analytic expression into a system of equations for a grid with 6 points using central differences.
#
# <em>Write your answer by hand using paper/tablet and include the image below.</em>
#     
#

# %% [markdown]
# _Your answer here._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1b:</b>
#
# Write the system of equations by hand for a grid with 6 points. In other words, construct the matrix A and vectors T and b.
#
# <em>Write your answer by hand using paper/tablet and include the image below.</em>
#     
#

# %% [markdown]
# _Your answer here._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1c:</b>
#
# Implement the discretized system of equations in a code cell.
#
# <em>We have already done this for you! Your task is to read the cell and make sure you understand how the matrices are implemented. Reading the code should help you formulate the matrices in Task 2.1b.</em>
#     
#

# %% [markdown]
# _Add your image here._

# %%
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

# %% [markdown]
#
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2:</b>
#
# This task will adapt the problem from 2.1 to incorporate Neumann boundary conditions in three steps: a) writing the new matrix by hand, b) adapting the code from 2.1c, c) reflecting on what this represents physically.
#
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2a:</b>
#
# Write the system of equations by hand for a grid with 6 points, incorporating the Neumann condition.
#
# Approximate the Neuman boundary $\frac{dT}{dx}=0$ by using the Backward difference for first order differential equation of first order accuracy.
#
# <em>Write your answer by hand using paper/tablet and include the image below.</em>
#     
#

# %% [markdown]
# _Your answer here._

# %% [markdown]
#
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2b:</b>
#
# Now adapt the code from Task 2.1c and revise it to incorporate the Neumann boundary condition.
#
# <em>Copy and past the code from 2.1c below, then modify it.</em>
#
# </p>
# </div>

# %%
YOUR_CODE_HERE

# %% [markdown]
#
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2c:</b>
#
# Reflect on the difference between the problem solved in Task 2.1 in comparison to 2.2. How are we changing the physics of the problem being solved by changing the boundary condition? What does this mean in reality for the temperature distribution in the bar over time?
#
# </p>
# </div>

# %% [markdown]
#

# %%
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

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3:</b>
#
# Just as we did in Task 2.1, this task has three parts: a) discretize the analytic expression into a system of equations using <b>forward differences</b>, b) write the system of equations by hand (e.g., the matrices), and c) implement the discretized system of equations in a code cell.
#
# Here we focus on <b>Dirichlet</b> conditions again.
# </div>    
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3a:</b>
#
# Discretize the analytic expression into a system of equations for a grid with 6 points using <b>forward differences</b>.
#
# <em>Write your answer by hand using paper/tablet and include the image below.</em>
#     
#

# %% [markdown]
# _Your answer here._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3b:</b>
#
# Write the system of equations by hand for a grid with 6 points. In other words, construct the matrix A and vectors T and b.
#
# <em>Write your answer by hand using paper/tablet and include the image below.</em>
#     
#

# %% [markdown]
# _Your answer here._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%;vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3c:</b>
#
# Implement the discretized system of equations in a code cell.
#
# <b>This time we did not do it for you!</b> Copy the code from Task 2.1c and revise it to solve the system of equations using <b>Forward Differences</b>. Keep the Dirichlet conditions.
#     
#

# %%
YOUR_CODE_HERE


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.4:</b>
#
# How much finer does your grid has to be in the forward difference implementation to get a similar value at x = 0.02 as in the central difference implementation? Vary your dx.
#
#
# </p>
# </div>

# %% [markdown]
# _Your answer here._

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; width:95%; border-radius: 10px">
# <p>
# <b>Bonus Task</b> 
#     
# The matrix inversion using numpy is one way to solve the system, another is the <code>gauss_jordan</code> method, written below, and another one is the sparse matrix-based method in the cell afterwards. Here, we will just have a brief comparison to see how these solvers perform when the matrix is large. Change <code>dx</code> to 0.0002 of the original code that solves the second degree ODE and test the time it takes by each method.
#     
# </p>
# </div>

# %%
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


# %%
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

# %% [markdown]
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png"/>
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png"/>
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
