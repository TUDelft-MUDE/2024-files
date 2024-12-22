# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------------------
# Creating an array using lists
A = np.array([[1, 1], 
              [2, 2], 
              [3, 3], 
              [4, 4]])
A.shape

# ----------------------------------------
scale = np.array([YOUR_CODE_HERE])

# ----------------------------------------
A = A.T
scale @ A

# ----------------------------------------
A = YOUR_CODE_HERE

# ----------------------------------------
plt.plot(A[0, :], A[1, :], 'ko') # Same as plt.plot(*A)
plt.show()

# ----------------------------------------
empty = np.zeros(shape=(2, 2))
shear = np.ones(shape=(2, 2))

line_A_x = np.linspace(0, 8, num=10)
line_A = np.array([line_A_x, 5*line_A_x + 2])

print(empty)
print()
print(shear)
print()
print(line_A)

# ----------------------------------------
# Generate scale through a different method
identity = np.eye(N = A.shape[0])

# Check it's really identity
truth_array = identity @ A == A # This is actually an array with some boolean values
assert (identity @ A == A).all()

# Making scale again:
scale_2 = 2 * identity
assert (scale_2 == scale).all()

# ----------------------------------------
reflect = np.array([[0, 1], [1, 0]])
result = YOUR_CODE_HERE

plt.plot(*result, "or")
plt.plot(*line_A, "ob")
plt.show()

# ----------------------------------------
data_x = np.linspace(0, 100, num=100)
data_y = data_x * 5 + 2 + np.random.random(size = (100,))

def fit_a_line_to_data(data_x, data_y):
    A = np.array([data_x, np.ones(len(data_x))]).T
    [slope, intercept], _, _, _ = np.linalg.lstsq(A, data_y, rcond=None)
    return slope, intercept

def fit_a_line_to_data_2(data_x, data_y):
    ## Complete the function here ##
    A = np.array([YOUR_CODE_HERE, np.ones(len(data_x))]).T
    [slope, intercept] = np.linalg.solve(A.T @ A, YOUR_CODE_HERE)
    return slope, intercept

plt.plot(data_x, data_y, "ok")

slope1, intercept1 = fit_a_line_to_data(data_x, data_y)
slope2, intercept2 = fit_a_line_to_data_2(data_x, data_y)

plt.plot(data_x, slope1*data_x + intercept1, "b")
plt.plot(data_x, slope2*data_x + intercept2, "r")
plt.show()

# ----------------------------------------
N = 1000
A = np.random.random(size = (2, N))
plt.plot(*A, "ok")
plt.show()

# ----------------------------------------
A = np.random.normal(YOUR_CODE_HERE, YOUR_CODE_HERE, size = (2, N))
plt.plot(*A, "ok")
plt.show()

# ----------------------------------------
X = norm(loc = 0, scale = 1)
print(X.stats())

# ----------------------------------------
p_x_lt_0 = YOUR_CODE_HERE

# ----------------------------------------
p_x_gt_1 = YOUR_CODE_HERE

# ----------------------------------------
x = np.linspace(-10, 10, num=1000)
plt.plot(YOUR_CODE_HERE, YOUR_CODE_HERE)
plt.show()

# ----------------------------------------
def create_sample(N):
    "Create N samples each of height and width."
    height = np.array(norm.rvs(YOUR_CODE_HERE))
    width = np.array(norm.rvs(YOUR_CODE_HERE))
    return height, width

def compute_area(height, width):
    "Compute the area of the rectangle."
    return YOUR_CODE_HERE

def area_mean_std(area_data):
    "Find the mean and std dev of the area."
    area_mean = YOUR_CODE_HERE
    area_std = YOUR_CODE_HERE
    return area_mean, area_std

def plot_data_and_pdf(data, mean, std):
    "Compare the histogram of data to a normal pdf defined by mean and std."
    histogram_data = plt.hist(data, bins = 10,
                              density=True, stacked=True,  edgecolor='black', linewidth=1.2)
    x = np.linspace(min(histogram_data[1]), max(histogram_data[1]), num=1000)
    area_norm = norm(loc=mean, scale=std)
    plt.plot(x, YOUR_CODE_HERE, color='red', linewidth=2.0)
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

# ----------------------------------------
N = 500
height, width = create_sample(N)
area = compute_area(height, width)
area_mean, area_std = area_mean_std(area)
plot_data_and_pdf(area, area_mean, area_std)

