# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
# from vis import *
from tickets import *
from minutes import *
from models import *

# %%
t = Tickets()
t.add([[1,60]])


# %%
m = Models(model_id=2)
prob = m.pickle_picker('load', 0)
exp_t = m.pickle_picker('load', 1)



# %%
def compute_expectation(p, E_N_w,
                        W=300000, ticket_cost=3):

    ticket_cost = 3
    N_t = len(p)
    I = ticket_cost*N_t

    expected_winnings = p*W
    expected_winnings_shared = expected_winnings/(E_N_w + 1)
    expected_net_benefit = expected_winnings_shared - ticket_cost

    print(f"Expected winnings: \t\t{np.sum(expected_winnings):10.2e}")
    print(f"Expected winnings shared: \t{np.sum(expected_winnings_shared):10.2e}")
    print(f"Expected net benefit: \t\t{np.sum(expected_net_benefit):10.2e}")
    print(f"Benefit:Cost ratio B/C: \t{np.sum(expected_winnings_shared)/I:10.3f}")
    print(f"  no. where B/C > 1: \t\t{int(np.sum(expected_winnings_shared > ticket_cost))}")
    print(f"  no. where B/C < 1: \t\t{int(np.sum(expected_winnings_shared < ticket_cost))}")

    return expected_net_benefit

# %%

# expected_winnings = prob*m.winnings
# expected_winnings_shared = expected_winnings/(exp_t + 1)
# expected_benefit = expected_winnings_shared - m.cost

# print(f"Expected winnings: \t\t{np.sum(expected_winnings):10.2e}")
# print(f"Expected winnings shared: \t{np.sum(expected_winnings_shared):10.2e}")
# print(f"Expected benefit: \t\t{np.sum(expected_benefit):10.2e}")

# m.plot(expected_benefit)


enb = compute_expectation(prob, exp_t);


# %%
def summarize_array(arr, name):
    summary = {
        'mean': np.nanmean(arr),
        'min': np.nanmin(arr),
        'max': np.nanmax(arr),
        'num_zeros': np.sum(arr == 0),
        'num_nans_or_infs': np.sum(np.isnan(arr) | np.isinf(arr)),
        'sum': np.nansum(arr)
    }
    print(f"Summary for {name}:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3e}")
        else:
            print(f"{key}: {value}")
    print("\n")

summarize_array(prob, 'prob')
summarize_array(m.winnings, 'm.winnings')
summarize_array(exp_t, 'exp_t')
summarize_array(enb, 'enb')

# %%
type(t.tickets[4])

# %%
t = Tickets()
t.add([[15,50], [6, 22]])
t.status()

# %%
t_p_and_n = np.zeros((t.N(), 2))
for i in range(t.N()):
    t_p_and_n[i, 0] = m.get_p([t.tickets[i]])
    t_p_and_n[i, 1] = m.get_expected_tickets([t.tickets[i]])

# %%
enb = compute_expectation(t_p_and_n[:, 0], t_p_and_n[:, 1])
enb_sparse = Models.data_zeros(t.tickets, enb)
m.plot(enb_sparse)

# %% [markdown]
# **End of notebook.**
#
# <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
#   <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#   </div>
#   <div style="font-size: 75%; margin-top: 10px; text-align: right;">
#     By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
#     &copy; 2024 TU Delft. 
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
#     <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
