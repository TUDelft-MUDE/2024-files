import numpy as np
import matplotlib.pyplot as plt

distance, temperature = np.genfromtxt("auxiliary_files/data_2.csv", skip_header = 1, delimiter=",", unpack=True)

temperature.size

temperature_is_nan = np.isnan(temperature)

print("The first 10 values are:", temperature_is_nan[0:10])
print(f"There are {temperature_is_nan.sum()} NaNs in array temperature")

temperature = temperature[~temperature_is_nan]

temperature.size

distance.size == temperature.size

distance = distance[~temperature_is_nan]
distance.size==temperature.size

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

print(temperature.min())
print(temperature.max())

distance = distance[temperature!=-999]
temperature = temperature[temperature!=-999]

print(distance.size == temperature.size)
temperature.size

mask = temperature!=999
distance = distance[mask]
temperature = temperature[mask]

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

distance = distance[np.nonzero(temperature)]
temperature = temperature[np.nonzero(temperature)]

distance = distance[temperature<50]
temperature = temperature[temperature<50]

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

temperature = np.where(temperature > 15, temperature, temperature * 1.5)

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

mean_temperature = temperature.mean()
print(f"{mean_temperature = :.3f}")

variance_temperature = temperature.var()
print(f"{variance_temperature = :.3f}")

print('hello world')

import platform
print("Running on a", platform.system(),"machine named:", platform.node())

