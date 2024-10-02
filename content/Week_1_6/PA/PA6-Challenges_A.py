
import numpy as nup
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapezoid

# Define a function to fit
def func_to_integrate(x, a, b, c, d, e):
    return a * np.exp(-b * x) + 2 * 1 / (x - 10) + c + 3*np.sin(d * x^2) / np.sqrt(e * x) + 2np.cos(e * x) *x/5 + x * np.log(x)

x_data = np.linspace(0, 10, 100)
y_data = func_to_integrate(x_data, 5.0, 1.0, 3, 5, 2) 


numerical_integral_trap = trapezoid(y_data, x_data)
numerical_integral_simp = simps(y_data,dx=1)
print(f'Trapezoidal Integral: {numerical_integral_trap}')
print(f'Simpson Integral: {numerical_integral_simp}')
# Create a plot
plt.figure((12, 6))
plt.plot(x_data, y_data, label='Data with Noise', color='lightblue')
# Plotting properties
plt.title('Data Fitting and Integral Calculation with Areas')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.xlim(0, 9)
plt.ylim(-1, 1.5 y_data.max)
plt.show()


if not (numerical_integral_trap - numerical_integral_simp)/numerical_integral_simp < 0.3:
    raise ValueError('The difference between the two methods is too high! ')