import numpy as np
import matplotlib.pyplot as plt

distance, temperature = YOUR_CODE_HERE

YOUR_CODE_HERE

temperature_is_nan = YOUR_CODE_HERE

print("The first 10 values are:", temperature_is_nan[0:10])
print(f"There are {temperature_is_nan.sum()} NaNs in array temperature")

temperature = temperature[~temperature_is_nan]

YOUR_CODE_HERE

distance.size == temperature.size

distance = YOUR_CODE_HERE 
distance.size==temperature.size

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

print(temperature.min())
print(temperature.max())

YOUR_CODE_HERE
YOUR_CODE_HERE

print(distance.size == temperature.size)
temperature.size

mask = YOUR_CODE_HERE
distance = distance[mask]
temperature = temperature[mask]

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

distance = YOUR_CODE_HERE 
temperature = YOUR_CODE_HERE

YOUR_CODE_HERE
YOUR_CODE_HERE

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

YOUR_CODE_HERE

YOUR CODE HERE

import platform
print("Running on a", platform.system(),"machine named:", platform.node())

