---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
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

```python


mesh = Mesh(coordinates, length)
mesh.define_triangles()
mesh.get_all_sides();
```

```python
mesh.try_triangles()
```

```python
mesh.refine_mesh()
mesh.plot_triangles()
```
