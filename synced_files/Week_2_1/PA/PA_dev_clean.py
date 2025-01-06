%load_ext autoreload
%autoreload 2
import numpy as np
from city import *
coordinates = np.array(
              [[7.000, 1.000],
               [4.000, 2.732],
               [3.000, 1.000],
               [5.000, 1.000],
               [3.000, 4.464],
               [1.000, 4.464],
               [6.000, 2.732],
               [1.000, 1.000],
               [7.000, 4.464],
               [2.000, 2.732],
               [5.000, 4.464]])
my_plan = Plan(coordinates)
my_plan.define_triangles()
my_plan.get_all_sides();
my_plan.try_triangles()
my_plan.refine_mesh()
my_plan.plot_coordinates();
my_plan = Plan(coordinates)
my_plan.plot_coordinates();
print(my_plan.side_length)
triangles = [[7, 9, 8]]
my_plan = Plan(coordinates)
my_plan.try_triangles([[7, 9, 2],
                       [9, 1, 2]], triangle_id=range(2))
my_plan.define_triangles()
my_plan.plot_triangles()
print(len(my_plan.triangles))
my_plan.triangles
my_plan.define_shared_sides()
print(my_plan.shared_sides)
my_plan.plot_shared_sides([my_plan.shared_sides[2]])
my_plan.get_all_sides()
len([[2, 3, 5]])
triangles = [[7, 9, 2],
             [9, 1, 2],
             [1, 3, 6],
             [3, 6, 0],
             [5, 9, 4],
             [9, 4, 1],
             [4, 1, 10],
             [1, 10, 6],
             [10, 6, 8],
             [1, 2, 3]]
my_plan = Plan(coordinates, triangles)
my_plan.plot_triangle(range(10))
my_plan.check_triangles()
sides = [[[9, 2], [0, 8]]]
my_plan.define_shared_sides(sides)
my_plan.plot_shared_sides(range(len(sides)));
my_plan.get_kapsalon_coordinates()
my_plan.get_bar_coordinates()
my_plan.plot_everything();
x=1
assert (
    x==1), (
        'cool')
import numpy as np
import sys
np.set_printoptions(precision=3,
                    threshold=sys.maxsize,
                    floatmode='fixed')
x = np.array([[1.312323,2.,3.],[1.,2.,3.]])
print(x)
