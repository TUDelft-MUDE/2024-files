<userStyle>Normal</userStyle>

somehhow this was used to create the pkl file `List_of_radial_object_all_minutes.pkl` or `List_of_radial_object.pkl`

```python
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tickets import *
from minutes import *
from models import *
```

```python

```

```python
counts_2018 = np.genfromtxt('2018_count_250108_0209.csv', delimiter=',', skip_header=1)
day_min_counts = Minutes.day_min(counts_2018[:, 0])
print(f"Loaded {len(counts_2018)} rows of data")
print(f"The shape is {day_min_counts.shape}")

tickets_2018 = np.genfromtxt('2018_tickets_250108_0209.csv', delimiter=',', skip_header=1)
day_min_tickets = Minutes.day_min(tickets_2018)
print(f"Loaded {len(day_min_tickets)} rows of data")
print(f"The shape is {tickets_2018.shape}")
print()
print(f"The mean day is {np.mean(day_min_tickets[:, 0])}")
print(f"The std day is {np.std(day_min_tickets[:, 0])}")
print(f"The mean min is {np.mean(day_min_tickets[:, 1])}")
print(f"The std min is {np.std(day_min_tickets[:, 1])}")
```

```python
all = Tickets()
all.add([[1, 60]])
day_min_all = Minutes.day_min(all.tickets)

parameters = [np.mean(day_min_tickets[:, 0]),
              np.std(day_min_tickets[:, 0]),
              np.mean(day_min_tickets[:, 1]),
              np.std(day_min_tickets[:, 1])]

transform = Minutes.get_transform(parameters)

day_c, min_c = transform(day_min_counts[:,0], day_min_counts[:,1])
radius_c = Minutes.radius(day_c, min_c)

day_a, min_a = transform(day_min_all[:,0], day_min_all[:,1])
radius_a = Minutes.radius(day_a, min_a)

radius_max = np.max(radius_a)

print(f"Maximum radius is: {radius_max:.3f}")
```

```python
def get_band_distribution(minute,
                          radius_c, counts_2018, radius_a, transform,
                          std_increment=0.125):

    day_c, min_c = Minutes.get_day_min(minute[0])
    day_t, min_t = transform(day_c, min_c)
    radius = Minutes.radius(day_t, min_t)
    
    if isinstance(std_increment, list):
        assert len(std_increment) == 2, "std_increment list must have two values"
        radius_min = radius - std_increment[0]
        radius_max = radius + std_increment[1]
    elif isinstance(std_increment, float):
        radius_min = radius - std_increment
        radius_max = radius + std_increment
    else:
        raise ValueError("std_increment must be a list of length 2 or a float")
    
    if radius_min < 0:
        d_radius = radius_max - radius_min
        radius_min = 0
        radius_max = d_radius

    if radius_max > max(radius_a):
        d_radius = radius_max - radius_min
        radius_min = radius_min - d_radius
        radius_max = max(radius_a)
        
    d = evaluate_ticket_dist_i(radius_min, radius_max,
                             radius_c, counts_2018[:, 1],
                             radius_a)
    return d
```

```python
# minute = Minutes.get_minutes([['April'], 25, 13, 10])
minute = Minutes.get_minutes([['April'], 29, 13, 30])
print(f"Minute: {minute[0]}")

d = get_band_distribution(minute,
                          radius_c, counts_2018, radius_a, transform,
                          std_increment=[.2, .2])
d.hist();
```

```python
minutes = np.arange(0,60*1440).tolist()

dist_all_minutes = []

for minute in minutes:

    dist_all_minutes.append(
        get_band_distribution(
            [minute], radius_c, counts_2018,
            radius_a, transform, std_increment=0.125
            )
            )

```

```python
with open('pickles/List_of_radial_object_all_minutes.pkl', 'wb') as f:
    pickle.dump(dist_all_minutes, f)

print("RadialDist list saved to 'List_of_radial_object_all_minutes.pkl'")
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
    By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
    &copy; 2024 TU Delft. 
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
    <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
