import matplotlib.pyplot as plt
import numpy as np
from utilities_solution import *

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

mesh.plot_triangles(triangle_id=4) # useful for identifying the triangle id
mesh.set_initial_conditions(default=20,
                            special_triangles=[[4, 40]])
mesh.plot_triangles(fill_color='initial_conditions');

mesh.solve(20, 100, 50)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);

mesh.refine_mesh();
mesh.plot_triangles();

mesh.solve(20, 100, 50)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);

side_length= 5.0
t_final=20
D=50
Nt_vec=np.arange(200, 650, 50, dtype=int)
Stability = []
max_temp = []
for Nt in Nt_vec:
    _,stability,temp=mesh.solve(20, Nt=Nt, D=D)
    Stability.append(stability)
    max_temp.append(temp.max())
    print("==================================")
plt.figure(figsize=(10, 5))
plt.plot(Nt_vec, Stability, 'o-')
for x, y, temp in zip(Nt_vec, Stability, max_temp):
    plt.text(x, y,f'$T=${temp:.1e} C', fontsize=6, ha='center', va='top')
plt.xlabel('Number of time steps')
plt.ylabel(r'D*$\Delta t$ Surface/Volume/centroid_distance)')
plt.title('Stability criteria vs Number of time steps ')
plt.axhline(0.5, color='r', linestyle='--')
plt.grid()
plt.show()

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

