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
import numpy as np
import matplotlib.pyplot as plt


# %%

x = np.array([0, 36, 62, 96])
y = np.array([0, 3e5, 3.2e6, 1e5])

x *= 15
y /= 3.2e6
print(x)
print(y)

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
m = Models(model_id=1)
m.plot(1)

# %%
max(scaled_minutes)

# %%
expected_tickets, scale_day, scale_minute = m.ticket_model_1(m.ticket_model_dict)

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Scaled minutes subplot
minutes = np.arange(0, 1441)
scaled_minutes = [scale_minute(minute) for minute in minutes]
axs[0].plot(minutes, scaled_minutes)
axs[0].set_title('Scaled Minutes')
axs[0].set_xlabel('Minutes')
axs[0].set_ylabel('Scaled Value')

# Scaled days subplot
days = range(61)
scaled_days = [scale_day(day) for day in days]
axs[1].plot(days, scaled_days)
axs[1].set_title('Scaled Days')
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Scaled Value')


# Set y-limits for all subplots
for ax in axs:
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
print(expected_tickets(33, 863))
print(m.get_expected_tickets(33, 863))

# %%
reference_day = 33
reference_min = 863

fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Expected minutes subplot for varying minutes
expected_tickets_for_minutes = [expected_tickets(reference_day, minute) for minute in minutes]
axs[0].plot(minutes, expected_tickets_for_minutes)
axs[0].set_title('Expected Minutes for Varying Minutes')
axs[0].set_xlabel('Minutes')
axs[0].set_ylabel('Expected Minutes')

# Expected minutes subplot for varying days
expected_tickets_for_days = [expected_tickets(day, reference_min) for day in days]
axs[1].plot(days, expected_tickets_for_days)
axs[1].set_title('Expected Minutes for Varying Days')
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Expected Minutes')

# Expected minutes subplot for decreasing minutes by 30 each day
decreasing_minutes = [reference_min - 30 * (day - reference_day) for day in days]
expected_tickets_for_decreasing_minutes = [expected_tickets(day, minute) for day, minute in zip(days, decreasing_minutes)]
axs[2].plot(days, expected_tickets_for_decreasing_minutes)
axs[2].set_title('Expected Minutes for Decreasing Minutes by 30 Each Day')
axs[2].set_xlabel('Days')
axs[2].set_ylabel('Expected Minutes')

plt.tight_layout()
plt.show()

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
