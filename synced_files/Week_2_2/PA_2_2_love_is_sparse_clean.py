import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import timeit
def create_dense(size: int, percentage: float) -> np.array:
    matrix = YOUR_CODE_HERE
    matrix[YOUR_CODE_HERE] = 0
    return matrix
test_size = YOUR_CODE_HERE
test_percentage = YOUR_CODE_HERE
matrix = create_dense(test_size, test_percentage)
assert np.count_nonzero(matrix) < test_percentage*1.1*test_size**2
my_matrix_size = YOUR_CODE_HERE
assert my_matrix_size == 8*test_size**2
csr_matrix = YOUR_CODE_HERE
bsr_matrix = YOUR_CODE_HERE
print(f"CSR matrix size: {csr_matrix.data.size} bytes")
print(f"Compared to the normal matrix, CSR uses this fraction of space: {csr_matrix.data.nbytes/my_matrix_size:0.3f}")
print(f"BSR matrix size: {bsr_matrix.data.size} bytes")
print(f"Compared to the normal matrix, BSR uses this fraction of space: {bsr_matrix.data.nbytes/my_matrix_size:0.3f}")
blank = np.zeros(shape=(4, 4))
blueprint = np.array([[0, 0.5], 
                      [1, 0.5]])
for i in range(2):
    blank[np.ix_([i*2, i*2 + 1], [1, 2])] = blueprint
print(blank)
N = 1000
relationship = np.zeros(shape=(N, N))
YOUR_CODE_HERE
N_ITS = 1000
T = 5 # Seconds
dt = T/N_ITS
def test(rel_matrix):
    state = np.zeros(N); state[0] = 1
    for i in range(N_ITS):
        state = state + rel_matrix @ state * dt
csr_matrix = YOUR_CODE_HERE
bsr_matrix = YOUR_CODE_HERE
print(f"Standard: {timeit.timeit('test(relationship)', globals=globals(), number=10)/10:.4f}")
print(f"CSR: {timeit.timeit('test(csr_matrix)', globals=globals(), number=10)/10:.4f}")
print(f"BSR: {timeit.timeit('test(bsr_matrix)', globals=globals(), number=10)/10:.4f}")
