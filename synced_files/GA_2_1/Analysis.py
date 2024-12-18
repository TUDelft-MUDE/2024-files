# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"
# # GA 2.1: FVM with an Unstructured  Mesh (diffusion)
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1. For: 15 November, 2024.*

# %% [markdown]
# # Overview
#
# This assignment considers an interesting 2D shape: the "U" of MUDE! We will use the diffusion equation to compute the distribution of temperature in the U subject to specific boundary conditions and initial conditions.
#
# This assignment contains three parts:
#
# 1. expressing the diffusion equation for triangle volumes and formulating algebraic equations with the finite volume method,
# 2. implementing and solving this method in an unstructured orthogonal mesh,
# 3. analyzing potential downsides to the discretized approach used if applied to non-orthogonal meshes, and considering how they might be corrected.
#
# Remember, even though the problem is 2D, the numerical scheme will be fromulated by treating _volumes_ (as opposed to dimensions $x$ and $y$ directly).

# %% [markdown]
# ## Part 1: Using Just Your Hands
#
# The diffusion equation expressed in its reduced form is:
#
# $$
# \frac{\partial \phi}{\partial t} = \nabla \cdot D \nabla \phi
# $$
#
# Note that for triangle shape volumes, the fluxes are not necessarily directed in the $x$ direction. Rather, in the normal direction to the surfaces of each volume, as the flux direction is given by the $\phi$ gradients and the information propagates in all directions.
#
# Over the next tasks, you will write step by step its derivation.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1</b>
#
# Integrate the PDE over triangle volumes of interest. 
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.2</b>
#
# Transform the corresponding volume integral into a surface integral using Gauss's theorem.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3</b>
#
# Approximate the integrals using numerical integration, using the midpoint rule. Write explicitly the three surface fluxes.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.4</b>
#
# Divide both sides of the equation by $\Delta V$. The "depth" of the volume is common with the surface flux area. Write down the resulting equation.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.5</b>
#
# Approximate the derivatives! Use central differences in space and Forward Euler in time. The subindices can be defined by arbitrarily naming the volume of interest and the adjacent volumes.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.5</b>
#
# The equation you derived above describes volumes with <b>three</b> interior sides. For a volume with <b>one</b> side being an exterior one, modify the equation above to implement the following Neumann condition (which replaces the discretized gradient term):
#
# $$
# \frac{\partial \phi }{\partial n} = 10 [C^o/m]
# $$
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# ## Part 2: Implementation!
#
# Below, the coordinates of the triangle vertices that cover the domain of interest are defined. The boundary conditions are also specified and incorporated in the resulting object `mesh` once the class `Mesh` is instantiated. The class `Mesh` is defined in `utilities.py` and has an identical structure and similar functions as provided in your PA for this week, which define key characteristics of the mesh and volumes, as well as providing useful plotting and solving methods.
#
# Note: the `Mesh` class and it's usage is illustrated extensively in the companion notebook `mesh_tips.ipynb`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from utilities import *

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.0</b>
#
# Execute the cell below and study the resulting geometry until you are comfortable recognizing the problem that will be solved. In particular, check that you can identify which boundary conditions are applied, and where, before moving on.
#
# If you are not sure how to interpret the code or figures, refer to <code>mesh_tips.ipynb</code>.
#
# </p>
# </div>

# %%
length = 10

coordinates = [[  0.,    0.   ],
               [ 10.,    0.   ],
               [  5.,   -8.660],
               [ 15.,   -8.660],
               [ 10.,  -17.320],
               [ 20.,  -17.320],
               [ 25.,   -8.660],
               [ 30.,  -17.320],
               [ 35.,   -8.660],
               [ 30.,    0.   ],
               [ 40.,    0.   ]]

coordinates = np.array(coordinates)

boundaries = [[['Neumann', +1], [0, 1]],
              [['Neumann',  0], [1, 3, 6, 9]],
              [['Neumann', -1], [9, 10]],
              [['Neumann',  0], [10, 7, 4, 0]]]

mesh = Mesh(coordinates, length, boundaries)
mesh.plot_triangles();
mesh.plot_boundaries();

# %% [markdown]
# Before we continue to finding the solution, let's explore the relationship of today's mesh with the bars and kapsalon shops we considered in the PA for this week.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1</b>
#
# Write down the similarities between the housing-bar-kapsalon problem (PA 2.1) and the FVM representation of the diffusion equation here. Specifically, the locations and quantities associated with each of the three things.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2</b>
#
# Suppose **just for this single task** that the FDM would be used, sketch the potential grid points and draw the boundaries that would represent the same domain in $x$ and $y$ as defined by <code>coordinates</code> in the previous task. 
#
# This is to illustrate one of the contrasts between FDM and FVM.
#
# </p>
# </div>

# %% [markdown]
# Write your answer here.

# %% [markdown]
# Now we will **continue with code implementation** by adding some information to the object that defines our problem, `mesh`. Remember to refer to `mesh_tips.ipynb` if you are not sure how to use it!

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3</b>
#
# First, use the method <code>set_initial_temperature()</code> to define the initial conditions for the volumes. The initial temperature should be 20 degrees everywhere except at the volume in the middle, where the temperature is 40 degrees.  
#
# </p>
# </div>

# %%
mesh.plot_triangles(YOUR_CODE_HERE) # useful for identifying the triangle id
mesh.set_initial_conditions(YOUR_CODE_HERE)
mesh.plot_triangles(fill_color='initial_conditions');

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.4</b>
#
# **Solve**.
#
# Using the <code>solve</code> method, solve the problem for conditions where $t_{final}=20$, $N_t=100$ and $D=50$.
#
# Use the resulting plot to see if you reach the solution you expect. Remember to consider the boundary conditions that were defined for you as well, not only the initial conditions.
#
# </p>
# </div>

# %%
mesh.solve(YOUR_CODE_HERE, YOUR_CODE_HERE, YOUR_CODE_HERE)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.5</b>
#
# **What's going on?!**
#
# In the previous task, you should have realized that the solution is _not_ what we expect.
#
# It turns out we screwed up! In the massive file <code>utilities.py</code>, there is </b>one</b> line of code that needs to be fixed in order to solve this problem. Can you find it and fix it?
#
# The first group to find the solution will win a prize!
#
# <em>Hint: it's not in the plotting method, and is related to something you derived in the tasks above.</em>
#
# <b>Note:</b> you should expect to spend some time reading the code (at least one very specific part of the code); not only will this help you fix the problem, but it will be very useful for answering the first few questions in the Report.
#
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.6</b>
#
# **Refine**
#
# Refine the mesh and create a plot to check visually how the geometry has changed.
#
# </p>
# </div>

# %%
YOUR_CODE_HERE
mesh.plot_triangles();

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.7</b>
#
# **Solve**.
#
# Solve the problem for the new mesh using the same time and diffusion parameters as before.
#
# <em>Hint: copy/paste the same code from when you solved it the first time!</em>
#
# </p>
# </div>

# %%
YOUR_CODE_HERE

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.8</b>
#
# **Broken again?!**
#
# The solution for the refined mesh did not work---this time it is not a problem with the code, but another issue. See if you can make some adjustments and fix the solution!
#
# Note that you will be asked about stability in the Report, so you might as well calculate this and record the values now.
#
# <em>Tip: even though the mesh has smaller volumes (triangles), the solution should look similar to the previous (unrefined) mesh.</em>
#
# </p>
# </div>

# %% [markdown]
# ## Part 3: Evaluating the implementation for non-equilateral triangles 
#
# Computations in meshes with non-equilateral triangles have added error sources that would need to be corrected to have an accurate solution. In this section you will analyse and reflect on the potential downsides of your implementation for non-equilateral triangle volumes by looking only at the fluxes between two volumes. The file `utilities.py` and rest of the code in this notebook is completely irrelevant to this Part (except the `numpy` and `matplotlib` import).

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.1:</b>
#
# The vertex coordinates of the first triangle to be analyzed are: [0,0] , [0,1] , [1,0]. The second triangle share the first two vertices and have coordinates [0,0] , [0,1] , [-1,0]. 
#
# Plot the vertices and the triangles edges.
#
# </p>
# </div>

# %%
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 0 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

plt.scatter(x_v1, y_v1)
plt.scatter(x_v2, y_v2)

def plotting_volumes(coordinates):
    for i in range(3):
        start = coordinates[i]
        end = coordinates[(i+1)%3]
        plt.plot([start[0], end[0]], [start[1], end[1]],
            color='black', linestyle='--', linewidth=2)

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    



# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.2:</b>
#
# Plot the centroid of each triangle, its connecting line and the middle distance. For the scheme you implemented in the previous task, write down what this implies for the **numerical integration approximation of the flux at the surface**.
#
# Specifically, consider the location where the flux is calculated, as implied by the numerical scheme, as well as the integration that is implied by the analytic surface integral over the surface of the shared triangle side.
# </p>
# </div>

# %%
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 0 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.3:</b>
#
# Now lets analyze a different situation and change the coordinates of the second triangle to [0,0] , [0,1] , [-1,1].
#
# Plot the centroid of each triangle, its connecting line and the middle distance. For the numerical scheme you implemented, write the implication for the **numerical derivative approximation of the flux at the surface**.
#
# Specifically, consider the _direction_ in which the flux is calculated, as implied by the numerical scheme, as well as the quantity demanded by the analytic expression (i.e., the surface integral).
#
# </p>
# </div>

# %%
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 1 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    

# %% [markdown]
# Write your answer here.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.4:</b>
#
# Now lets analyze one last situation and change the coordinates of the second triangle to [0,0] , [0,1] , [-1.5,1].
#
# Plot the centroid of each triangle, its connecting line and the middle distance. Write down what this implies for the **numerical derivative approximation of the flux normal to the surface**. 
#
#
# </p>
# </div>

# %%
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1.5 ]
y_v2 = [ 0 , 1 , 1 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    

# %% [markdown]
# Write your answer here.

# %% [markdown]
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
