import matplotlib.pyplot as plt
import numpy as np
from utilities import *

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

print('Boundary sides:', mesh.boundary_sides)
print('Boundary types:', mesh.boundary_types)
print(f'Boundary side 2 is defined for side '
      +f'{mesh.boundary_sides[2]}')
print(f'Side {mesh.boundary_sides[2]} is defined by '
      +f'coordinates {mesh.all_sides[mesh.boundary_sides[2]]}')

mesh.plot_boundaries();

mesh.initial_conditions

mesh.set_initial_conditions(default=5, special_triangles=[[2, 3],[5,9]])
mesh.plot_triangles(fill_color='initial_conditions', show_labels=False);

mesh.plot_triangles(triangle_id=[2, 5], fill_color='initial_conditions', show_labels=False);

print(mesh.triangles)
mesh.plot_coordinates();

mesh.solve(20, 1000, 50);

print('First time step:', mesh.unknowns[0,:])
print('Last time step:', mesh.unknowns[-1,:])

mesh.plot_triangles(fill_color='unknowns', show_labels=False);

mesh.plot_triangles(fill_color='unknowns',
                    time_step=5,
                    show_labels=False);

mesh.refine_mesh();
mesh.plot_triangles();

mesh.solve(20, 1000, 50);
mesh.plot_triangles(fill_color='unknowns', show_labels=False);

mesh.refine_mesh();
mesh.plot_triangles();

