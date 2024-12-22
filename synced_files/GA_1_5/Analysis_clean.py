# ----------------------------------------
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# ----------------------------------------
data=pd.read_csv(filepath_or_buffer='justIce.csv',index_col=0)
data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

plt.figure(figsize=(15,4))
plt.scatter(data.index,data, color='green', marker='x')
plt.xlabel('Year')
plt.ylabel('Ice thickness [cm]')
plt.grid()

# ----------------------------------------
data_2021 = data.loc['2021']

plt.figure(figsize=(15,4))
plt.scatter(data_2021.index,data_2021, color='green', marker='x')
plt.xlabel('Date')
plt.ylabel('Ice thickness [cm]')
plt.grid()


# ----------------------------------------
h_ice = (data_2021.to_numpy()).ravel()
t_days = ((data_2021.index - data_2021.index[0]).days).to_numpy()

dh_dt_FD = YOUR_CODE_HERE

# ----------------------------------------
fig, ax1 = plt.subplots(figsize=(15,4))

ax1.scatter(YOUR_CODE_HERE, dh_dt_FD,
            color='blue', marker='o', label='dh_dt_FD Forward Difference')\

ax1.set_xlabel('Days')
ax1.set_ylabel('Growth Rate [cm/day]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()


ax2 = ax1.twinx()
ax2.scatter(t_days, h_ice,
            color='green', marker='x', label='h_ice Measurements')
ax2.set_ylabel('Ice thickness [cm]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc='upper left')

plt.show()


# ----------------------------------------
dh_dt_BD = YOUR_CODE_HERE

fig, ax1 = plt.subplots(figsize=(15,4))

ax1.scatter(YOUR_CODE_HERE, dh_dt_FD,
            color='blue', marker='o', label='dh_dt_FD Forward Difference')
ax1.scatter(YOUR_CODE_HERE, dh_dt_BD,
            color='red', marker='o', label='dh_dt_BD Backward Difference')

ax1.set_xlabel('Days')
ax1.set_ylabel('Growth Rate [cm/day]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()

ax2 = ax1.twinx()
ax2.scatter(t_days, h_ice,
            color='green', marker='x', label='h_ice Measurements')
ax2.set_ylabel('Ice thickness [cm]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc='upper left')

plt.show()

# ----------------------------------------
dh_dt_CD = YOUR_CODE_HERE

fig, ax1 = plt.subplots(figsize=(15,4))

ax1.scatter(YOUR_CODE_HERE, dh_dt_FD,
            color='blue', marker='o', label='dh_dt_FD Forward Difference')
ax1.scatter(YOUR_CODE_HERE, dh_dt_BD,
            color='red', marker='o', label='dh_dt_BE Backward Difference')
ax1.scatter(YOUR_CODE_HERE, dh_dt_CD,
            color='purple', marker='o', label='dh_dt_CD Central Difference')

ax1.set_xlabel('Days')
ax1.set_ylabel('Growth Rate [cm/day]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()

ax2 = ax1.twinx()
ax2.scatter(t_days, h_ice, color='green', marker='x', label='h_ice Measurements')
ax2.set_ylabel('Ice thickness [cm]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc='upper left')

plt.show()

# ----------------------------------------
x = np.linspace(-3*np.pi, 5*np.pi, 400)

def f(x):
    return YOUR_CODE_HERE

plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE, color='b', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Plot of $f(x) = 2cos(x) + sin(x)$");

# ----------------------------------------
def f_1(x):
    return YOUR_CODE_HERE

def f_2(x):
    return YOUR_CODE_HERE

def f_3(x):
    return YOUR_CODE_HERE

def f_4(x):
    return YOUR_CODE_HERE

# ----------------------------------------
x0 = YOUR_CODE_HERE
taylor_1 = YOUR_CODE_HERE
taylor_2 = YOUR_CODE_HERE
taylor_3 = YOUR_CODE_HERE
taylor_4 = YOUR_CODE_HERE

# ----------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         label='$f(x) = 2cos(x) + sin(x)$', color='b', linewidth=2)
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         label='First Order', linestyle='--', color='g', linewidth=2)
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         label='Second Order', linestyle='--', color='r', linewidth=2)
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         label='Third Order', linestyle='--', color='m', linewidth=2)
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
         label='Fourth Order', linestyle='--', color='y', linewidth=2)


plt.scatter([x0], [f(x0)],
            color='k', marker='o', label=f'Expansion Point ($x = {x0:0.3f}$)')

plt.xlabel('x')
plt.ylabel('$f(x)$')
plt.title('Taylor Series Expansion of $f(x) = 2cos(x) + sin(x)$')
plt.legend()
plt.xlim(-1,9)
plt.ylim(-10,10)

plt.grid(True)
plt.show();

# ----------------------------------------
error_1 = YOUR_CODE_HERE
error_2 = YOUR_CODE_HERE
error_3 = YOUR_CODE_HERE
error_4 = YOUR_CODE_HERE

plt.figure(figsize=(10, 4))
plt.plot(x, error_1,
         label='First Order', color='g', linewidth=2)
plt.plot(x, error_2,
         label='Second Order', color='r', linewidth=2)
plt.plot(x, error_3,
         label='Third Order', color='m', linewidth=2)
plt.plot(x, error_4,
         label='Fourth Order', color='y', linewidth=2)

plt.xlabel('x')
plt.ylabel('Absolute Error: $f(x)$-Taylor Order')
plt.title('Absolute Error of Taylor Series Approximations')
plt.xlim(np.pi-1,np.pi+1)
plt.ylim(0,0.01)
plt.legend()

plt.grid(True)
plt.show();

# ----------------------------------------
def f2D(x, y):
    return YOUR_CODE_HERE

x0 = YOUR_CODE_HERE
y0 = YOUR_CODE_HERE

def taylor2D(x, y):
    return (YOUR_CODE_HERE)

# ----------------------------------------
# Create a meshgrid of x and y values
x = np.linspace(-2+x0, 2+x0, 100)
y = np.linspace(-2+y0, 2+y0, 100)
X, Y = np.meshgrid(x, y)

# Calculate the original function values and the approximation
Z_orig = f2D(X, Y)
Z_approx = taylor2D(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_orig, rstride=1, cstride=1, cmap='Reds',
                label='Original Function')
ax.plot_surface(X, Y, Z_approx, rstride=1, cstride=1, cmap='Blues',
                alpha=0.7, label='Taylor Approximation')

# Set labels and legend
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.legend()

# Show the plot
plt.title('Original Function vs. Taylor Approximation')
plt.show()


# ----------------------------------------
error_2d = YOUR_CODE_HERE

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, error_2d, cmap='Reds', label='Absolute Error')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Absolute Error')
plt.title('Absolute Error between $f(x, y)$ and Taylor Approximation')
plt.show()


