# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %% [markdown]

# %% [markdown]

# %%

A = np.array([[1, 1], 
              [2, 2], 
              [3, 3], 
              [4, 4]])
A.shape

# %% [markdown]

# %%

scale = np.array([[2, 0], [0, 2]])

# %% [markdown]

# %%

A = A.T
scale @ A

# %% [markdown]

# %% [markdown]

# %%

A = A.T

# %% [markdown]

# %%
plt.plot(A[0, :], A[1, :], 'ko') 
plt.show()

# %% [markdown]

# %%
empty = np.zeros(shape=(2, 2))
shear = np.ones(shape=(2, 2))

line_A_x = np.linspace(0, 8, num=10)
line_A = np.array([line_A_x, 5*line_A_x + 2])

print(empty)
print()
print(shear)
print()
print(line_A)

# %% [markdown]

# %% [markdown]

# %%

identity = np.eye(N = A.shape[0])

truth_array = identity @ A == A 
assert (identity @ A == A).all()

scale_2 = 2 * identity
assert (scale_2 == scale).all()

# %% [markdown]

# %% [markdown]

# %%

reflect = np.array([[0, 1], [1, 0]])
result = reflect @ line_A

plt.plot(*result, "or")
plt.plot(*line_A, "ob")
plt.show()

# %% [markdown]

# %%
<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>Minor note (warning). Don't use the CSS danger/warning blocks because they don't display when converting to HTML with pandoc (won't display right on course files on website).</p></div>

# %% [markdown]

# %% [markdown]

# %%
data_x = np.linspace(0, 100, num=100)
data_y = data_x * 5 + 2 + np.random.random(size = (100,))

def fit_a_line_to_data(data_x, data_y):
    A = np.array([data_x, np.ones(len(data_x))]).T
    [slope, intercept], _, _, _ = np.linalg.lstsq(A, data_y, rcond=None)
    return slope, intercept

def fit_a_line_to_data_2(data_x, data_y):
    
    
    A = np.array([data_x, np.ones(len(data_x))]).T
    [slope, intercept] = np.linalg.solve(A.T @ A, A.T @ data_y)
    
    return slope, intercept

plt.plot(data_x, data_y, "ok")

slope1, intercept1 = fit_a_line_to_data(data_x, data_y)
slope2, intercept2 = fit_a_line_to_data_2(data_x, data_y)

plt.plot(data_x, slope1*data_x + intercept1, "b")
plt.plot(data_x, slope2*data_x + intercept2, "r")
plt.show()

# %% [markdown]

# %% [markdown]

# %%
N = 1000
A = np.random.random(size = (2, N))
plt.plot(*A, "ok")
plt.show()

# %% [markdown]

# %% [markdown]

# %%

A = np.random.normal(loc = 10, scale = 5, size = (2, N))
plt.plot(*A, "ok")
plt.show()

# %% [markdown]

# %%
X = norm(loc = 0, scale = 1)
print(X.stats())

# %% [markdown]

# %%

p_x_lt_0 = X.cdf(0)

# %% [markdown]

# %%

p_x_gt_1 = 1 - X.cdf(1)

# %% [markdown]

# %%

x = np.linspace(-10, 10, num=1000)
plt.plot(x, X.pdf(x))
plt.show()

# %% [markdown]

# %% [markdown]

# %%
def create_sample(N):
    "Create N samples each of height and width."
    
    
    
    height = np.array(norm.rvs(loc = 5000, scale = 5, size=N))
    width = np.array(norm.rvs(loc = 2000, scale = 10, size=N))
    return height, width

def compute_area(height, width):
    "Compute the area of the rectangle."
    
    
    return height*width

def area_mean_std(area_data):
    "Find the mean and std dev of the area."
    
    
    area_mean = np.mean(area_data)
    area_std = np.std(area_data)
    
    
    
    return area_mean, area_std

def plot_data_and_pdf(data, mean, std):
    "Compare the histogram of data to a normal pdf defined by mean and std."
    histogram_data = plt.hist(data, bins = 10,
                              density=True, stacked=True,  edgecolor='black', linewidth=1.2)
    x = np.linspace(min(histogram_data[1]), max(histogram_data[1]), num=1000)
    area_norm = norm(loc=mean, scale=std)
    
    
    
    
    
    plt.plot(x, area_norm.pdf(x), color='red', linewidth=2.0)
    plt.title("Comparison of Normal PDF to Histogram of Data")
    plt.xlabel("Area of Rectangle [m^2]")
    plt.ylabel("Probability Density (PDF) and Frequency (data)")
    plt.show()

# %% [markdown]

# %%
N = 500
height, width = create_sample(N)
area = compute_area(height, width)
area_mean, area_std = area_mean_std(area)
plot_data_and_pdf(area, area_mean, area_std)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

