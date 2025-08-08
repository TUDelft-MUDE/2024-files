%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tools import *
day_dist = stats.norm(loc=33, scale=6.5)
p_day =day_dist.cdf(33) - day_dist.cdf(32)
print(f"P[day = historical average] = {p_day:.4f}")
min_dist = stats.norm(loc=863, scale=190)
p_min = min_dist.cdf(863) - min_dist.cdf(862)
print(f"P[min = historical average] = {p_min:.4f}")
p_day_and_min = p_day*p_min
print(f"P[day and min] = {p_day_and_min:.4f}")
print(f"P[day = historical average] = {p_day:.2e}")
print(f"P[min = historical average] = {p_min:.2e}")
print(f"P[day and min] = {p_day_and_min:3e}")
unlikely_day = 16
unlikely_min = 3*60 + 33
p_unlikely_day = day_dist.cdf(unlikely_day+0.5) - day_dist.cdf(unlikely_day-0.5)
p_unlikely_min = min_dist.cdf(unlikely_min+0.5) - min_dist.cdf(unlikely_min-0.5)
p_unlikely_day_and_min = p_unlikely_day*p_unlikely_min
print(f"P[unlikely day] = {p_unlikely_day:.2e}")
print(f"P[unlikely min] = {p_unlikely_min:.2e}")
print(f"P[unlikely day and min] = {p_unlikely_day_and_min:.2e}")
t = Tickets()
t.add([34])
t.status()
t.show()
t.add([[20, 23]])
t.status()
t.show()
t.add([[37], [13, 16]])
t.status()
t.show()
t.add([[3, 8], [13, 16]])
t.status()
t.show()
t.add([[5, 15], [3, 6], [15, 45]])
t.status()
t.show()
t.add([[38, 40, 42, 49, 56],
       [3, 6, 9, 12, 15],
       [15, 16, 17, 18, 19, 20, 30, 40]])
t.status()
t.show()
m = Models(model_id=2)
t_test = Tickets()
t_test.add([[25], [13], [0]])
p_ticket = m.get_p(t_test.tickets)
print(f"The probability of the ticket is {p_ticket:.3e}")
t_test.status()
t_test.show()
m = Models()
prob_T = np.zeros((len(t.tickets)))
for i, ti in enumerate(t.tickets):
    prob_T[i] = m.get_p([ti])
prob_T_matrix = m.map_data_to_day_min(prob_T, t.tickets)
m.plot(prob_T_matrix,
       custom_title="Probability of ticket",
       custom_label="Probability of ticket")
prob_any_T_wins = np.sum(prob_T)
print(f"Prob of any ticket winning: {prob_any_T_wins:.3e}")
m.plot(0)
