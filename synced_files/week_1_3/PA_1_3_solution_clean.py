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
distance, temperature = np.genfromtxt("auxiliary_files/data_2.csv", skip_header = 1, delimiter=",", unpack=True)

# %% [markdown]

# %%
temperature.size

# %% [markdown]

# %%
temperature_is_nan = np.isnan(temperature)

print("The first 10 values are:", temperature_is_nan[0:10])
print(f"There are {temperature_is_nan.sum()} NaNs in array temperature")

# %% [markdown]

# %%
temperature = temperature[~temperature_is_nan]

# %% [markdown]

# %%
temperature.size

# %% [markdown]

# %%
distance.size == temperature.size

# %% [markdown]

# %% [markdown]

# %%
distance = distance[~temperature_is_nan]
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
distance = distance[temperature!=-999]
temperature = temperature[temperature!=-999]

# %% [markdown]

# %%
print(distance.size == temperature.size)
temperature.size

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
mask = temperature!=999
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
distance = distance[np.nonzero(temperature)]
temperature = temperature[np.nonzero(temperature)]

# %% [markdown]

# %% [markdown]

# %%
distance = distance[temperature<50]
temperature = temperature[temperature<50]

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
mean_temperature = temperature.mean()
print(f"{mean_temperature = :.3f}")

variance_temperature = temperature.var()
print(f"{variance_temperature = :.3f}")

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
print('hello world')

# %% [markdown]

# %% [markdown]

# %%
import platform
print("Running on a", platform.system(),"machine named:", platform.node())

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

