%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tickets import *
from minutes import *
from models import *
m = Models(model_id=0)
m.plot(1)
expected_tickets, scale_day, scale_minute = m.ticket_model_0(m.ticket_model_dict)
reference_day = m.ticket_model_dict["references"][0]
reference_min = m.ticket_model_dict["references"][1]
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
minutes = np.arange(0, 1441)
scaled_minutes = [scale_minute(minute) for minute in minutes]
axs[0].plot(minutes, scaled_minutes)
axs[0].set_title('Scaled Minutes')
axs[0].set_xlabel('Minutes')
axs[0].set_ylabel('Scaled Value')
days = np.arange(0, 61)
scaled_days = [scale_day(day) for day in days]
axs[1].plot(days, scaled_days)
axs[1].set_title('Scaled Days')
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Scaled Value')
for ax in axs:
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
print(expected_tickets(33, 863))
print(m.get_expected_tickets(33, 863))
fig, axs = plt.subplots(3, 1, figsize=(10, 18))
expected_tickets_for_minutes = [expected_tickets(reference_day, minute) for minute in minutes]
axs[0].plot(minutes, expected_tickets_for_minutes)
axs[0].set_title('Expected Minutes for Varying Minutes')
axs[0].set_xlabel('Minutes')
axs[0].set_ylabel('Expected Minutes')
expected_tickets_for_days = [expected_tickets(day, reference_min) for day in days]
axs[1].plot(days, expected_tickets_for_days)
axs[1].set_title('Expected Minutes for Varying Days')
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Expected Minutes')
decreasing_minutes = [reference_min - 30 * (day - reference_day) for day in days]
expected_tickets_for_decreasing_minutes = [expected_tickets(day, minute) for day, minute in zip(days, decreasing_minutes)]
axs[2].plot(days, expected_tickets_for_decreasing_minutes)
axs[2].set_title('Expected Minutes for Decreasing Minutes by 30 Each Day')
axs[2].set_xlabel('Days')
axs[2].set_ylabel('Expected Minutes')
plt.tight_layout()
plt.show()
