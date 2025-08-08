%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
print(Minutes.get_day_hour_min(0))
Minutes.get_day_hour_min(42450)
t = Tickets()
t.add([['April', 'May'], [25, 5], [2, 5, 7], [0, 20]])
t.add([1], True)
t.add(['April', 27], True)
t.show()
t.add([5, [2, 8]])
t.show()
t.show()
