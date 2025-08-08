import numpy as np
import matplotlib.pyplot as plt
C = 0.01
alpha = 2
pf_societal = C/10**alpha
print(pf_societal)
N_values = np.array([1, 1000])
limit_line = C/N_values**alpha
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(N_values, limit_line, color='blue', markersize=10, label='Limit Line')
ax.plot(10, pf_societal, 'ro', markersize=10, label='Factory Spill')
ax.set_title('Societal Risk Limit for Sickness from Spill', fontsize=16)
ax.set_xlabel('Number of Sick People, N [$-$]', fontsize=14)
ax.set_ylabel('Failure probability, per year', fontsize=14)
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
ax.grid(True)
plt.show()
n_and_p = np.array([[1,   0.099],
                    [2,   7e-4],
                    [5,   2e-4],
                    [10,  9e-5],
                    [20,  5e-6],
                    [60,  3e-6],
                    [80,  6e-7],
                    [100,  4e-7],
                    [150, 8e-8]])
N_plot = n_and_p[:, 0]
p_for_0 = 1 - n_and_p[:, 1].sum()
print(f"The probability of 0 sick people is {p_for_0:.2f}")
p_cumulative = np.cumsum(n_and_p[:, 1]) + p_for_0
fig, ax = plt.subplots(figsize=(8, 6))
ax.step(N_plot, 1-p_cumulative, where='post',
        color='green', markersize=10, label='Cumulative Risk')
ax.plot(N_values, limit_line, color='blue', markersize=10, label='Limit Line')
ax.plot(10, pf_societal, 'ro', markersize=10, label='Factory Spill')
ax.set_title('Cumulative Societal Risk for Sickness from Spill', fontsize=16)
ax.set_xlabel('Number of Sick People, N [$-$]', fontsize=14)
ax.set_ylabel('Cumulative probability, per year', fontsize=14)
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.8, 1200)
plt.ylim(1e-8, 1)
ax.grid(True)
plt.show()
n_and_p_modified = n_and_p.copy()
n_and_p_modified[5, 0] = 40
N_plot = n_and_p_modified[:, 0]
p_cumulative_modified = np.cumsum(n_and_p_modified[:, 1]) + p_for_0
fig, ax = plt.subplots(figsize=(8, 6))
ax.step(N_plot, 1-p_cumulative_modified, where='post',
        color='green', markersize=10, label='Cumulative Risk')
ax.plot(N_values, limit_line, color='blue', markersize=10, label='Limit Line')
ax.plot(10, pf_societal, 'ro', markersize=10, label='Factory Spill')
ax.set_title('Cumulative Societal Risk for Sickness from Spill', fontsize=16)
ax.set_xlabel('Number of Sick People, N [$-$]', fontsize=14)
ax.set_ylabel('Cumulative probability, per year', fontsize=14)
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.8, 1200)
plt.ylim(1e-8, 1)
ax.grid(True)
plt.show()
expected_value = np.sum(n_and_p[:, 0]*n_and_p[:, 1])
print(f"The expected number of sick people is {expected_value:.2f} per year")
expected_value_modified = np.sum(n_and_p_modified[:, 0]*n_and_p_modified[:, 1])
print("After making modifications:")
print(f"    expected number of sick people is {expected_value_modified:.2f} per year")
