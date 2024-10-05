## Some small bugs for you to find!
import numpy as np
import matplotlib.pylab as plt

# Part 1
# ======
# Create a matrix with 5's on the diagonal and 1's on the diagonal
# below the main diagonal
A = np.zeros((5, 5))
np.fill_diagonal(A, 5)
A[range(3), range(1, 5)] = 1

"""SOLUTION

Traceback excerpt:
==================
A[np.arange(3), np.arange(1, 5)] = 1
IndexError: shape mismatch: indexing arrays could not be broadcast
together with shapes (3,) (4,)

Explanation:
============
IndexError indicates that we have indexed the matrix wrong, but its
not clear why. The first guess might be that we are referring to
an index that is out of bounds, but that is not the case here. It
turns out that the problem is that the two arrays we are creating with
np.arange are of different lengths. The first array has length 3, and
the second array has length 4. As the indices are used to change the
values in the matrix, they need to be of the same length. To fix this,
the two arange arrays should be of the same length. Because we know
the lower diagonal is being set and the matrix is 5x5, we need to
change 4 values, so arange(4) should be used instead of arange(3).

Solution:
=========
Change index from 4 to 3 so each range() has same number of elements:
    A[range(3), range(1, 5)] = 1
"""

# Part 2
# ======
# We want to compute the exponent of x = 0, 4, 8, 12, 16, 20 
x = range(0, 21, 4)
y = np.exp(x)

# then we want to change the first value of x to 1 instead of 0
x[0] = 1
y = np.exp(x)

assert x[0]==1, "The first value of x should be 1"
assert y[0]==np.exp(1), "The first value of y should be exp(1)"

"""SOLUTION

Traceback excerpt:
==================
    x[0] = 1
TypeError: 'range' object does not support item assignment

Explanation:
============
The error message mentions "item assignment", which is what we are
trying to do: assign the value 1 to the first _item_ in x. The mistake
is that we are doing this as if x is an Numpy array, but it is clearly
a range object. The fix is simple: define x using np.arange instead of
range.

If the line of code were more complex (e.g., included more terms and
variable assignments), it might not have been apparent which object
was the range, so a good debugging strategy would be to check the type
of each object.

Solution:
=========
Define x as an array with np.arange(); first line is thus:
    x = np.arange(0, 21, 4)
"""


# Part 3
# ======
# Some arbitrary script to prepare arrays for plotting
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = np.linspace(0, 10, 11)

print(a.size())

for i in range(0, a.size):
    a[i] = a[i] +a[i-1]

c = a+b

"""SOLUTION

Traceback excerpt:
==================
1. AttributeError: 'list' object has no attribute 'size'
2. TypeError: 'int' object is not callable
3. ValueError:  operands could not be broadcast together with 
                shapes (10,) (11,)

Explanation:
============
There are several issues in this one, all relatively straightforward to fix,

Solution:
=========

In order:
1:
    a = np.array(a)
2:
    a.size() --> a.size
3:
    np.linspace(0, 10, 11) --> np.arange(10)
    np.linspace(0, 10, 11) --> np.linspace(0, 10, 10)
"""