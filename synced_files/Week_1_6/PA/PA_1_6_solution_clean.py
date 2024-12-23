# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# help(plt.matshow)

# ----------------------------------------
A = [[1, 2, 1, 1],
     [2, 3, 2, 2],
     [1, 2, 1, 1],
     [1, 2, 1, 1]]

plt.matshow(A)
plt.show()

# ----------------------------------------
A = np.random.rand(100, 100)
plt.matshow(A)
plt.show()

# ----------------------------------------
# # help(range)
# 
# print('Part a:')
# [print(i) for i in YOUR_CODE_HERE]
# print('Part b:')
# [print(i) for i in YOUR_CODE_HERE]

# SOLUTION
print('Part 1:')
[print(i) for i in range(6)]

print('Part 2:')
[print(i) for i in range(2, 11, 2)]

# ----------------------------------------
# A = YOUR_CODE_HERE
# 
# assert np.all(A==1)
# assert A.shape==(3, 3)

# SOLUTION
A = np.ones((3,3))

assert np.all(A==1)
assert A.shape==(3, 3)

# ----------------------------------------
# A = YOUR_CODE_HERE
# np.YOUR_CODE_HERE
# 
# assert np.all(A.diagonal()==3)
# assert A.sum()==9
# assert A.shape==(3, 3)

# SOLUTION
A = np.zeros((3,3))
np.fill_diagonal(A, 3)

assert np.all(A.diagonal()==3)
assert A.sum()==9
assert A.shape==(3, 3)

# ----------------------------------------
# A = YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(10, 10)
# assert A.sum()==5
# assert np.sum(A==1)==5
# assert np.sum(A==0)==95

# SOLUTION
A = np.zeros((10,10))
A[range(1, 10, 2), range(1, 10, 2)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(10, 10)
assert A.sum()==5
assert np.sum(A==1)==5
assert np.sum(A==0)==95

# ----------------------------------------
# A = YOUR_CODE_HERE
# YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(5, 5)
# assert A.sum()==(5*5 + 2*4)

# SOLUTION
A = np.zeros((5, 5))
np.fill_diagonal(A, 5)
A[range(4), range(1, 5)] = 1
A[range(1, 5), range(4)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(5, 5)
assert A.sum()==(5*5 + 2*4)

# ----------------------------------------
# A = YOUR_CODE_HERE
# for i in YOUR_CODE_HERE:
#     YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(10, 10)
# assert A.sum()==25

# SOLUTION
A = np.zeros((10, 10))
for i in range(0, 10, 2):
    A[i, range(0, 10, 2)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(10, 10)
assert A.sum()==25

# ----------------------------------------
help(np.arange)

# ----------------------------------------
# x = YOUR_CODE_HERE
# print('Part 1:')
# [print(i) for i in YOUR_CODE_HERE]
# 
# assert type(x) == np.ndarray
# assert np.all(x == [0, 1, 2, 3, 4, 5])
# 
# x = YOUR_CODE_HERE
# print('Part 2:')
# [print(i) for i in YOUR_CODE_HERE]
# 
# assert type(x) == np.ndarray
# assert np.all(x == [2, 4, 6, 8, 10])

# SOLUTION
x = np.arange(6)
print('Part 1:')
[print(i) for i in x]

assert type(x) == np.ndarray
assert np.all(x == [0, 1, 2, 3, 4, 5])

x = np.arange(2, 11, 2)
print('Part 2:')
[print(i) for i in x]

assert type(x) == np.ndarray
assert np.all(x == [2, 4, 6, 8, 10])

