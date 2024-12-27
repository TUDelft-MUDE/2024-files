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

```python
length = 10
height = (length**2 - (length/2)**2)**0.5
x = [ 0 , length , length/2 , length/2+length ,   length  ,   2*length , (5/2)*length ,  3*length , (7/2)*length  ,  3*length , 4*length ]
y = [ 0 ,   0    , -height  ,     -height     , -2*height ,  -2*height , -height      , -2*height ,  -height      ,   0       , 0 ]
 
```

```python
import matplotlib.pyplot as plt

plt.scatter(x, y)
```

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
from utilities import *
```

```python
coordinates = np.array([x,y]).T
print(coordinates)
```

Boundaries are specified with a list of lists, where each item in the list defines a boundary in the format:

```
[coordinate_index, boundary_type]
```

Each boundary can be defined using as many nodes as desired, though the boundary will retain its shape regardless of how many times the mesh is refined (this is logical, as refinement only subdivides triangles, thus the sides of a triangle always remain straight).



```python
boundaries = [[['Neumann', +5], [0, 1]],
              [['Neumann',  0], [1, 3, 6, 9]],
              [['Neumann', +5], [9, 10]],
              [['Neumann',  0], [10, 7, 4, 0]]]
```

```python
mesh = Mesh(coordinates, length, boundaries)
# mesh.define_triangles()
# mesh.get_all_sides();
```

```python
mesh.triangles
```

```python
len(mesh.triangles)
```

```python
mesh.set_initial_conditions(default=5, special_triangles=[[2, 3],[5,9]])
mesh.plot_triangles(fill_color='initial_conditions');
```

```python
mesh.plot_boundaries();
```

```python
mesh.plot_boundary_sides()
```

```python
mesh.initial_conditions
```

```python
mesh.plot_triangles(fill_color='initial_conditions')
```

```python
mesh.try_triangles()
```

```python
# mesh.refine_mesh()
# mesh.get_initial_conditions()
mesh.plot_triangles(fill_color='initial_conditions');
```

```python
# mesh.set_initial_conditions(default=0, special_triangles=[[0, 10]])

mesh.solve(20, 1000, 50)
mesh.plot_triangles(fill_color='unknowns', time_step=-1,
                    show_labels=False);
# mesh.unknowns[50]
```

```python
mesh.refine_mesh();
mesh.plot_triangles();
```



```python
print(mesh.boundary_sides[0])
mesh.all_sides[mesh.boundary_sides[0]]
sorted([11,0])==sorted(mesh.all_sides[mesh.boundary_sides[0]])
len(mesh.boundary_side_types)

```
