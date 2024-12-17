---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
---

# Illustration `bivariate`



```python
import bivariate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvinecopulib as cop
import scipy.stats as st
```

```python
X_1 = st.norm(loc=3, scale=1)
X_2 = st.norm(loc=5, scale=1)

n = 10000

X_1_samples = X_1.rvs(size=n)
X_2_samples = X_2.rvs(size=n)

X_combined_samples = np.array([X_1_samples, X_2_samples]).T

X_class_A = bivariate.class_copula.Region_of_interest(
                random_samples=X_combined_samples)

X_class_A.plot_emperical_contours(bandwidth=4)

def underwater(X1,X2):
    Z_now = 10.0
    function = (Z_now - X1 - X2 <= 0)
    return function
  
X_class_A.function =  underwater
X_class_A.inside_function()
X_class_A.plot_inside_function();
```

```python
# define multivariate normal distribution

X = st.multivariate_normal(mean=[3, 5],
                           cov=[[1, 0.5],
                                [0.5, 1]])

n = 10000
X_samples = X.rvs(size=n)

X_class_A = bivariate.class_copula.Region_of_interest(
                random_samples=X_samples)

X_class_A.plot_emperical_contours(bandwidth=4)

def underwater(X1,X2):
    Z_now = 10.0
    function = (Z_now - X1 - X2 <= 0)
    return function
  
X_class_A.function =  underwater
X_class_A.inside_function()
X_class_A.plot_inside_function();

```
