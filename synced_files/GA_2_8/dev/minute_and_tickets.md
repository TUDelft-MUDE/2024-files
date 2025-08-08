<userStyle>Normal</userStyle>

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import matplotlib.pyplot as plt
# from vis import *
from tickets import *
from minutes import *
```

```python
print(Minutes.get_day_hour_min(0))
# print(t.tickets[0:5])
```

```python
Minutes.get_day_hour_min(42450)
```

```python
t = Tickets()
# t.status()
t.add([['April', 'May'], [25, 5], [2, 5, 7], [0, 20]])
t.add([1], True)
# t.status()
t.add(['April', 27], True)
# t.status()
# t.show()
t.show()
```

```python
t.add([5, [2, 8]])
t.show()
```

```python
t.show()
```

<!-- #region -->


```
reference_day = "April 1"
plot_probabilities
plot_tickets


<!-- #endregion -->

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
    By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
    &copy; 2024 TU Delft. 
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
    <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
