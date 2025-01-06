<userStyle>Normal</userStyle>

# Week 1.5: Programming Tutorial

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6. October 7, 2024.*

_This notebook was prepared by Berend Bouvy and used in an in-class demonstration on Monday._


## Objects

```python
import numpy as np
import matplotlib.pyplot as plt
from func import *
import timeit
```

```python
# import data 
data = np.loadtxt('data.txt')
x = data[:,0]
y = data[:,1]

```

```python
A = #TODO: create the matrix A
x_hat, y_hat = #TODO: solve the system of equations
print(f"x_hat = {x_hat}")
```

```python
# runtimes

funcs = [FD_1, FD_2, FD_3, FD_4]

assert np.allclose(funcs[0](x, y), funcs[1](x,y)), "FD_1 and FD_2 are not equal"
assert np.allclose(funcs[0](x, y), funcs[2](x,y)), "FD_1 and FD_3 are not equal"
assert np.allclose(funcs[0](x, y), funcs[3](x,y)), "FD_1 and FD_4 are not equal"

runtime = np.zeros(4)
for i in range(4):
    runtime[i] = timeit.timeit(lambda: funcs[i](x, y), number=1000)

print(f"runtimes: {runtime}")
```

<!-- #region -->
**End of notebook.**

<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
  <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
  </div>
  <div style="font-size: 75%; margin-top: 10px; text-align: right;">
    &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. 
    This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
