# ---

# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"

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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import matplotlib.pyplot as plt
import numpy as np
from utilities import *

# %% [markdown]

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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
mesh.plot_triangles(YOUR_CODE_HERE) 
mesh.set_initial_conditions(YOUR_CODE_HERE)
mesh.plot_triangles(fill_color='initial_conditions');

# %% [markdown]

# %%
mesh.solve(YOUR_CODE_HERE, YOUR_CODE_HERE, YOUR_CODE_HERE)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE
mesh.plot_triangles();

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %% [markdown]

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

# %% [markdown]

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

# %% [markdown]

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

# %% [markdown]

