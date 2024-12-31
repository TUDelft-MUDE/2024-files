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

There are going to be several quantities that must be computed,
queried or saved that are defined for each minute, but may need
to be accessed in a variety of ways. For example:
- probability of breakup happening at a given minute
    P(m) = P(day)*P(minute)
- expected number of tickets bought at a certain minute
    E[t] --> defined for each minute
- whether or not the ticket is selected at a given minute
    1 if selected, 0 if not

Each of these would be convenient to compute every 15 min, hour,
day, etc, to make calculations faster and provide summary info.



"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from minutes import *
from typing import Union
from datetime import datetime, timedelta
import pickle

class Probability:
    def __init__(self):
        self.distribution_day = stats.norm(33, 6.5)
        self.distribution_min = stats.norm(863, 190)
        self.pickle_jar = "pickles"
        self.pickles = ['breakup_prob_hist',
                        'expected_tickets',
                        'purchased_tickets']
        self.initialize()

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

    def plot(self, which_pickle:Union[int,str], increment:str='min'):

        breakup_prob_hist = self.pickle_picker('load',
                                               which_pickle)

        fig, ax = plt.subplots()
        
        
        if increment == 'min':
            pass
            # no longer calculated here
        elif increment == 'hour':
            pass
            # can implement once probability can be specified by hour too
            # --> need to adapt input values
        else:
            raise ValueError("increment must be 'min' or 'hour'")
        
        cax = ax.imshow(breakup_prob_hist.T, aspect='auto', cmap='viridis', origin='lower')
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

    def pickle_jar_check(self, auto_update:bool=False):
        status = {}
        for i in self.pickles:
            try:
                status[i] = self.pickle_picker('check', i)
            except:
                if auto_update:
                    status[i] = self.pickle_picker('save', i)
                else:
                    status[i] = None
        return status


    def initialize(self):
        return self.pickle_jar_check(auto_update=True)

    def pickle_picker(self, mode:str, pickle_name:Union[int,str]):
        
        if isinstance(pickle_name, int):
            pickle_name = self.pickles[pickle_name]

        if pickle_name == 'breakup_prob_hist':
            if mode == 'save':
                pickle_state = {
                    "day": self.distribution_day,
                    "min": self.distribution_min
                    }

                breakup_prob_hist = np.zeros((60, 1440))
                for i in range(60*1440):
                        day, min = Minutes.get_day_min(i)
                        breakup_prob_hist[day-1, min] = self.get_p(day, min)

                with open('pickles/breakup_prob_hist.pkl', 'wb') as f:
                    pickle.dump(pickle_state, f)
                    pickle.dump(breakup_prob_hist, f)

            elif mode == 'check':
                with open('pickles/breakup_prob_hist.pkl', 'rb') as f:
                    pickle_state = pickle.load(f)
                    self.distribution_day = pickle_state["day"]
                    self.distribution_min = pickle_state["min"]
                print(f"Pickle for {pickle_name} is up to date.")

            elif mode == 'load':
                with open('pickles/breakup_prob_hist.pkl', 'rb') as f:
                    _ = pickle.load(f)
                    breakup_prob_hist = pickle.load(f)
                return breakup_prob_hist