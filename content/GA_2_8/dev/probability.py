"""
Use convention from minutes.py, where first minute of April 1 is:
- April 1, 00:00:00-00:00:59
- overall minutes as a continuous variable is 0
- Python index is minutes[0]

This means that the probability of that minut is F_M(1)-F_M(0),
where F_M(m) is the cumulative distribution function of random
variable M (minutes).

These are the moments, rounded from 2.8 in 2023.
distribution_day = (33, 6.5)
distribution_min = (863, 190)

"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from minutes import *
from typing import Union
from datetime import datetime, timedelta

class Probability:
    def __init__(self):
        self.distribution_day = stats.norm(33, 6.5)
        self.distribution_min = stats.norm(863, 190)
        # self.

    def get_p(self, argument_1: list, argument_2: Union[list, None]=None):
        """evaluate probability
        
        if 1 argument provided, it's minutes as a list
        if 2 arguments provided, it's day and minute as lists
        """
        if argument_2 is None:
            for minute in argument_1:
                day, min = Minutes.get_day_min(minute)
        else:
            day = argument_1
            min = argument_2
        
        p_day = (self.distribution_day.cdf(day)
                 - self.distribution_day.cdf(day - 1))
        p_min = (self.distribution_min.cdf(min)
                 - self.distribution_min.cdf(min - 1))
        return p_day*p_min

    def plot_probability(self, increment:str='min'):
        fig, ax = plt.subplots()
        density_matrix = np.zeros((60, 1440))
        
        if increment == 'min':
            for i in range(60*1440):
                day, min = Minutes.get_day_min(i)
                density_matrix[day-1, min] = self.get_p(day, min)
        elif increment == 'hour':
            pass
            # can implement once probability can be specified by hour too
            # --> need to adapt input values
        else:
            raise ValueError("increment must be 'min' or 'hour'")
        
        cax = ax.imshow(density_matrix.T, aspect='auto', cmap='viridis', origin='lower')
        fig.colorbar(cax, ax=ax, label='Probability')
        
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 24*60)
        ax.set_xlabel('Day Number (April 1 is Day 0)')
        ax.set_ylabel('Minutes of the Day')
        ax.set_title('Probability of Each Minute')
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(6*60))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(60))
        import matplotlib.dates as mdates

        secax = ax.secondary_xaxis('top')
        start_date = datetime.strptime("April 1", "%B %d")
        date_labels = [start_date + timedelta(days=i) for i in range(60)]
        secax.set_xticks(range(0, 60, 10))
        secax.set_xticklabels([date.strftime("%B %d") for date in date_labels[::10]], rotation=45, ha='left')
        secax.set_xlabel('Date')
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        plt.show()

    # def plot_