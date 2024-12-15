# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
x = 0
assert x == 1

# %% [markdown]

# %%
y = 0
assert y != 1, YOUR_MESSAGE_HERE

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
squares = []
for i in range(10):
    squares.append(i**2)

print(squares)

# %% [markdown]

# %%
squares = [i ** 2 for i in range(10)]

print(squares)

# %% [markdown]

# %%
squares_dict = {}
for i in range(10):
    squares_dict[i] = i ** 2

print(squares_dict)

# %% [markdown]

# %%
squares_dict = {i: i ** 2 for i in range(10)}

print(squares_dict)

# %% [markdown]

# %%
my_list = [1, 2, 3, 4, 5]
new_list = []
print(new_list)
assert new_list == [0.5, 1.0, 1.5, 2.0, 2.5], "new_list values are not half of my_list!"

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
help(plt.bar)

# %% [markdown]

# %%
plt.bar([], [])

# %% [markdown]

# %% [markdown]

# %%
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=YOUR_CODE_HERE,
        align=YOUR_CODE_HERE,
        edgecolor='black')

# %% [markdown]

# %%
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=YOUR_CODE_HERE,
        align=YOUR_CODE_HERE,
        edgecolor='black')

# %% [markdown]

# %% [markdown]

# %%
fig, ax = plt.subplots(1,1,figsize = (8,6))
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6])
fig.savefig('my_figure.svg')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

