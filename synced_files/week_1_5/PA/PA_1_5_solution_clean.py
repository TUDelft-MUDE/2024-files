# ----------------------------------------
# x = 0
# assert x == 1

# SOLUTION
x = 1
assert x == 1

# ----------------------------------------
# y = 0
# assert y != 1, YOUR_MESSAGE_HERE

# SOLUTION
y = 0
assert y != 1, "y should not be 1 but it is"

# ----------------------------------------
# YOUR_CODE_HERE

# ----------------------------------------
squares = []
for i in range(10):
    squares.append(i**2)

print(squares)

# ----------------------------------------
squares = [i ** 2 for i in range(10)]

print(squares)

# ----------------------------------------
squares_dict = {}
for i in range(10):
    squares_dict[i] = i ** 2

print(squares_dict)

# ----------------------------------------
squares_dict = {i: i ** 2 for i in range(10)}

print(squares_dict)

# ----------------------------------------
# my_list = [1, 2, 3, 4, 5]
# new_list = []
# print(new_list)
# assert new_list == [0.5, 1.0, 1.5, 2.0, 2.5], "new_list values are not half of my_list!" 

# SOLUTION
my_list = [1, 2, 3, 4, 5]
new_list = [x/2 for x in my_list]
assert new_list == [0.5, 1.0, 1.5, 2.0, 2.5], "new_list values are not half of my_list!" 

# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
help(plt.bar)

# ----------------------------------------
# plt.bar([], [])

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6])

# ----------------------------------------
# plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
#         width=YOUR_CODE_HERE,
#         align=YOUR_CODE_HERE,
#         edgecolor='black')

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=1,
        align='edge',
        edgecolor='black')

# ----------------------------------------
# plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
#         width=YOUR_CODE_HERE,
#         align=YOUR_CODE_HERE,
#         edgecolor='black')

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=-1,
        align='edge',
        edgecolor='black')

# ----------------------------------------
fig, ax = plt.subplots(1,1,figsize = (8,6))
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6])
fig.savefig('my_figure.svg')


