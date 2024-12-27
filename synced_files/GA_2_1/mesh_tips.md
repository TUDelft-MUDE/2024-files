<userStyle>Normal</userStyle>

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
---

# GA 2.1: Mesh Tips

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1. For: 15 November, 2024.*

The purpose of this notebook is to illustrate how to use the class `Mesh` which is defined in the file `utilities.py`.

```python
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import numpy as np
from utilities import *
```

The cell below begins with the geometry that is used to form the "U" for GA 1.2 It is defined by an array of points. When instantiating the class, the boundaries are specified using a list of lists, where each item (list) in the list defines the conditions, then the coordinates, of a boundary. For example:

```
boundaries = [[['Neumann', +1], [0, 1]]]
```

defines a boundary condition on the line connecting the 0th and 1st coordinates (rows 0 and 1 of `coordinates`, where a Neumann condition with $\partial\phi/\partial n=1$. Note in the example below, that multiple coordinates can be used to define a boundary line.

When the class is instantiated, all boundary sides that are not specified are unassigned. When the discretized scheme is solved (later with method `solve`), these unspecified boundaries receive a no flux boundary by default. Each boundary can be defined using as many nodes as desired, though the boundary will retain its shape regardless of how many times the mesh is refined (this is logical, as refinement only subdivides triangles, thus the sides of a triangle always remain straight).

```python
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
```

## Boundary Conditions

The boundary conditions are automatically processed when the class is instantiated. In particular, the following attributes are used to easily identify and apply the boundary conditions in the `solve` method.

`boundary_sides` identifies the sides of a triangle where a boundary is applied, where a "side" is the index of the list `all_sides`, which is itself a list of lists, each of which defines the index of the vertices defining the side in `.coordinates`

`mesh.boundary_types` defines the boundary condition, specified with a list of lists, as described above.

```python
print('Boundary sides:', mesh.boundary_sides)
print('Boundary types:', mesh.boundary_types)
print(f'Boundary side 2 is defined for side '
      +f'{mesh.boundary_sides[2]}')
print(f'Side {mesh.boundary_sides[2]} is defined by '
      +f'coordinates {mesh.all_sides[mesh.boundary_sides[2]]}')
```

A method `plot_boundaries` can be used to visualize the the sides where boundary conditions are applied.

```python
mesh.plot_boundaries();
```

## Initial Conditions

Initial conditions are defined for each triangle and can be found with the attribute `initial_conditions`. Note that the first time the cell below is run it will indicate that this is stored as an array and instantiated with default values of $T=0$ C for each triangle.

```python
mesh.initial_conditions
```

More interesting initial conditions can be evaluated using the `set_initial_conditions` method with the following keyword arguments:

- `default` will set the value of every triangle to the value specified
- `special_triangles` will set the value of specific triangles to the value specified (and takes precedence over `default` value!)

The conditions for `special_triangles` are specified as a list of lists, where the inner list contains two values, the triangle index and the initial condition (e.g., Temperature).

For example, the cell below sets an initial value of 5 for all triangles except triangles 2 and 5, which have temperature 3 and 9, respectively.

Note in particular that the plotting function `plot_triangles` has been enabled with a keyword argument to shade the triangles according to their initial condition. The indices of the coordinates can also be removed for clarity by setting keyword argument `show_labels` to `False`.

```python
mesh.set_initial_conditions(default=5, special_triangles=[[2, 3],[5,9]])
mesh.plot_triangles(fill_color='initial_conditions', show_labels=False);
```

If you would like to easily find out what the number is of a given triangle, use the `plot_triangles` method with keyword argument `triangle_id` set to an integer or list of indices. Here is an example:

```python
mesh.plot_triangles(triangle_id=[2, 5], fill_color='initial_conditions', show_labels=False);
```

If you are trying to find the index of a particular triangle, it could be useful to print the `triangles` attribute and compare to the plot of the coordinates to identify their indices. See if you can confirm visually using the results below that triangle 9 is the one in the top right, with coordinates 8, 9 and 10.

```python
print(mesh.triangles)
mesh.plot_coordinates();
```

## Solve!

The `Mesh` class has been set up with a method `solve` to solve the algebraic system of equations for FVM. The input arguments required define the time integration scheme (final time $t_{final}$ and number of time steps $N_t$), as well as the diffusion coefficient, $D$.

```python
mesh.solve(20, 1000, 50);
```

As seen above, a short message is printed when solving is completed. The results are stored in the attribute `unknowns` which includes the solutions for unknowns $\phi_i^n$ for all time steps and triangles. 

```python
print('First time step:', mesh.unknowns[0,:])
print('Last time step:', mesh.unknowns[-1,:])
```

Note that you can also visualize the solution with the method `plot_triangles` and by setting the keyword argument `fill_color` to `unknowns`:

```python
mesh.plot_triangles(fill_color='unknowns', show_labels=False);
```

The method `plot_triangles` can visualize specific time steps using keyword argument `time_step` (an integer). The default value is the final time step: `time_step=-1`.

Here is an example that shows the 5th time step, when the temperature has already started to diffuse from the high temperature triangles to the lower temperature adjacent ones.

**Note that you will not be able to see this until you have "fixed" the mistake in `utilities.py`**

```python
mesh.plot_triangles(fill_color='unknowns',
                    time_step=5,
                    show_labels=False);
```

## Refine the mesh

Executing the method `refine_mesh` will create a new mesh that is filled with triangles that are half the size of the prior triangles. The class automatically carries out all tasks needed to facilitate solving this new geometry with _the same boundary conditions and initial conditions specified in the original geometry._

The example below refines the mesh then visualizes it.

```python
mesh.refine_mesh();
mesh.plot_triangles();
```

Note that the solution technique works as expected.

In this case a note is printed about the color bar, which warns that the scale has been adjusted in the plot. This is important to note if you are comparing two figures, as if the scale is changed the colors for specific triangles can no longer be compared. This happens if the solution plotted has temperature values outside the range of initial conditions, which are the limits used to create the default color scale. It happens when the boundary conditions result in a final temperature that changes significantly from the initial conditions (for example, imagine initial conditions of 0 C, with boundary conditions where the flux into the volume is positive - the final temperature will be higher than 0!).

```python
mesh.solve(20, 1000, 50);
mesh.plot_triangles(fill_color='unknowns', show_labels=False);
```

Note that the mesh can be refined repeatedly; however, as the code is not optimized for efficiency, it will start to take a long time if you use this method too many times. Note that the third refinement (so the fourth geometry) takes 20 seconds to create. And the solution will then take a _very_ long time!

**We recommend that you don't refine the mesh more than 3 times, and only try to solve this 4th geometry if you are able to let the solution run for a few minutes.**

```python
mesh.refine_mesh();
mesh.plot_triangles();
```

**End of notebook.**
<h2 style="height: 60px">
</h2>
<h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    
</h3>
<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
