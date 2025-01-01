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

on init set_ticket_model sets up a dict and will create a function
to modify, change dict value then run self.get_ticket_model()

"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from minutes import *
from typing import Union
from datetime import datetime, timedelta
import pickle

class Models:
    def __init__(self, ticket_model:str='dev'):
        self.winnings = 300000
        self.cost = 3
        self.moments_day = (33, 6.5)
        self.moments_min = (863, 190)
        self.distribution_day = stats.norm(self.moments_day[0],
                                           self.moments_day[1])
        self.distribution_min = stats.norm(self.moments_min[0],
                                           self.moments_min[1])
        self.set_ticket_model(model_id='dev')
        self.get_ticket_model()
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
    
    def set_ticket_model(self, model_id:str='dev'):
        if model_id == 'dev':
            dict = {
                "references": [self.moments_day[0],
                             self.moments_min[0]],
                "scaling": [0.1, 0.01],
                "limits": [0, 100],
                "thresholds": [0.02, 0.9],
                "static_method_name": "ticket_model_0",
                "static_method": self.ticket_model_0
                }
            self.ticket_model_dict = dict
        else:
            raise ValueError(f"Model ID {model_id} undefined.")
        
    def get_ticket_model(self):
        model_maker = self.ticket_model_dict["static_method"]
        ticket_model, _, _ = model_maker(self.ticket_model_dict)
        self.ticket_model = ticket_model

    def get_expected_tickets(self,
                             argument_1: list,
                             argument_2: Union[list, None]=None):

        if argument_2 is None:
            for minute in argument_1:
                day, min = Minutes.get_day_min(minute)
        else:
            day = argument_1
            min = argument_2

        return self.ticket_model(day, min)
        

    def plot(self, which_pickle:Union[int,str], increment:str='min'):

        data = self.pickle_picker('load', which_pickle)
        
        titles = ['Historic Probability of Breakup',
                  'Expected Tickets',
                  'Purchased Tickets']
        labels = ['Probability', 'Ticket Count', 'Ticket Count']

        if isinstance(which_pickle, str):
            which_pickle = self.pickles.index(which_pickle)

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
        
        cax = ax.imshow(data.T, aspect='auto', cmap='viridis', origin='lower')
        fig.colorbar(cax, ax=ax, label=labels[which_pickle])
        
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 24*60)
        ax.set_xlabel('Day Number (April 1 is Day 0)')
        ax.set_ylabel('Minutes of the Day')
        ax.set_title(titles[which_pickle])
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
            
        if pickle_name == 'expected_tickets':
            if mode == 'save':
                pickle_state = self.ticket_model_dict.copy()
                # function can't be pickled, so remove it
                pickle_state.pop("static_method")
                expected_tickets = np.zeros((60, 1440))
                for i in range(60*1440):
                    day, min = Minutes.get_day_min(i)
                    expected_tickets[day-1, min] = self.get_expected_tickets(day, min)
                print(f"Min and max expected tickets: {expected_tickets.min()}, {expected_tickets.max()}")
                with open('pickles/expected_tickets.pkl', 'wb') as f:
                    pickle.dump(pickle_state, f)
                    pickle.dump(expected_tickets, f)
            elif mode == 'check':
                with open('pickles/expected_tickets.pkl', 'rb') as f:
                    _ = pickle.load(f)
                print(f"Pickle for {pickle_name} is up to date.")
            elif mode == 'load':
                with open('pickles/expected_tickets.pkl', 'rb') as f:
                    _ = pickle.load(f)
                    expected_tickets = pickle.load(f)
                return expected_tickets


    @staticmethod
    def ticket_model_0(d: dict):
        
        reference_day = d["references"][0]
        reference_min = d["references"][1]
        ticket_limits = d["limits"]
        thresholds = d["thresholds"]


        def scale_day(day, scaling_factor=d["scaling"][0]):
            if day <= reference_day:
                return np.exp(-scaling_factor * (reference_day - day))
            else:
                return np.exp(-scaling_factor * (day - reference_day))
            
        def scale_minute(minute, scaling_factor=d["scaling"][1]):
            if minute <= reference_min:
                return np.exp(-scaling_factor * (reference_min - minute))
            else:
                return np.exp(-scaling_factor * (minute - reference_min))

        def expected_tickets(day, min):
            day_scale = scale_day(day)
            min_scale = scale_minute(min)
            scale = day_scale * min_scale
            if scale < thresholds[0]:
                return ticket_limits[0]
            elif scale > thresholds[1]:
                return ticket_limits[1]
            else:
                return int(round(scale * (ticket_limits[1] - ticket_limits[0]) + ticket_limits[0]))
            
        return expected_tickets, scale_day, scale_minute
            