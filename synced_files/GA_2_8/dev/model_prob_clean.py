%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
from models import *
Minutes.get_days(['April', [5,7,7]])
m = Models()
m.plot(0)
