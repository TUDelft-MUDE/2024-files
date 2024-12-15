import numpy as np
import matplotlib.pyplot as plt

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

my_plan = Plan(YOUR_CODE_HERE, YOUR_CODE_HERE)

YOUR_CODE_HERE

my_plan.try_triangles([[0, 1, 2]])

all_triangles = [YOUR_CODE_HERE]
my_plan.try_triangles(all_triangles)

YOUR_CODE_HERE
my_plan.plot_triangles();

my_plan.plot_shared_sides([[[9, 2], [3, 8]]]);

sides = [YOUR_CODE_HERE] # remember to use format [[[a, b], [c, d]], ... ]
my_plan.plot_shared_sides(sides);

YOUR_CODE_HERE
my_plan.plot_shared_sides();

my_plan.get_bar_coordinates()
my_plan.plot_everything();

my_plan.get_kapsalon_coordinates()
my_plan.plot_everything();

