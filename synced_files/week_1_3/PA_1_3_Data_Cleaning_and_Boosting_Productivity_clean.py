# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]

# %% [markdown]

# %%
distance, temperature = YOUR_CODE_HERE

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %%
temperature_is_nan = YOUR_CODE_HERE

print("The first 10 values are:", temperature_is_nan[0:10])
print(f"There are {temperature_is_nan.sum()} NaNs in array temperature")

# %% [markdown]

# %%
temperature = temperature[~temperature_is_nan]

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %%
distance.size == temperature.size

# %% [markdown]

# %% [markdown]

# %%
distance = YOUR_CODE_HERE 
distance.size==temperature.size

# %% [markdown]

# %%
plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# %% [markdown]

# %%
print(temperature.min())
print(temperature.max())

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE
YOUR_CODE_HERE

# %% [markdown]

# %%
print(distance.size == temperature.size)
temperature.size

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
mask = YOUR_CODE_HERE
distance = distance[mask]
temperature = temperature[mask]

# %% [markdown]

# %% [markdown]

# %%
plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# %% [markdown]

# %% [markdown]

# %%
distance = YOUR_CODE_HERE 
temperature = YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE
YOUR_CODE_HERE

# %% [markdown]

# %%
plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# %% [markdown]

# %% [markdown]

# %%
temperature = np.where(temperature > 15, temperature, temperature * 1.5)

# %% [markdown]

# %%
plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
YOUR CODE HERE

# %% [markdown]

# %% [markdown]

# %%
import platform
print("Running on a", platform.system(),"machine named:", platform.node())

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

