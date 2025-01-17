import numpy as np
import matplotlib.pyplot as plt
from array import array
from typing import Union
from datetime import datetime, timedelta
import random
import scipy.stats as stats
import pickle
from scipy.sparse import lil_matrix

class Minutes:
    def __init__(self,
                 name="no name",
                 reference="April 1",
                 days=60):
        self.name = name
        self.reference = reference
        self.days = days
        self.tickets = np.zeros((days, 1440))
        self.empty = np.zeros((days, 1440))
        self.cost = 3.0

    @staticmethod
    def get_list_minutes(minutes: Union[list, int])-> list:
        """minutes in an hour, for assembly into list"""
        if isinstance(minutes, int):
            assert 0 <= minutes < 60,\
                "Invalid input: minutes should be between 0 and 59"
            return [minutes]
        elif isinstance(minutes, list):
            assert all(0 <= minute < 60 for minute in minutes),\
                "Invalid input: minutes should be between 0 and 59"
            if len(minutes) == 2:
                return list(range(minutes[0], minutes[1] + 1))
            else:
                if len(minutes) == 3 and minutes[-1] == minutes[-2]:
                    minutes.pop(-1)
                return minutes
        else:
            raise ValueError(("Invalid input: minutes should be a "
                             +"list or an integer"))

    @staticmethod
    def get_list_hours(hours: Union[list, int])-> list:
        """hours in a day, for assembly into list"""
        if isinstance(hours, int):
            assert 0 <= hours < 24,\
                "Invalid input: hours should be between 0 and 23"
            return [hours]
        elif isinstance(hours, list):
            assert all(0 <= hour < 24 for hour in hours),\
                "Invalid input: hours should be between 0 and 23"
            if len(hours) == 2:
                return list(range(hours[0], hours[1] + 1))
            else:
                if len(hours) == 3 and hours[-1] == hours[-2]:
                    hours.pop(-1)
                return hours
        else:
            raise ValueError(("Invalid input: hours should be a "
                              +"list or an integer"))
    
    @staticmethod
    def get_list_days(days: Union[list, int])-> list:
        """days absolute, for assembly into list"""
        if isinstance(days, int):
            assert days > 0,\
                "Invalid input: days should be greater than 0."
            return [days]
        elif isinstance(days, list):
            assert all(day > 0 for day in days),\
                "Invalid input: days should be greater than 0."
            if len(days) == 2:
                return list(range(days[0], days[1] + 1))
            else:
                # first enforce that repeating the last day
                # forces non-inclusive
                if len(days)==3 and days[-1] == days[-2]:
                    days.pop(-1)
                return days
        else:
            raise ValueError(("Invalid input: days should be a "
                             +"list or an integer"))



    @staticmethod
    def combine_hours_minutes(hours: Union[list, int,  None],
                              minutes: Union[list, int, None])-> list:
        """assumes day 0, hour 0"""
        if minutes is None:
            minutes = list(range(60))
        else:
            assert isinstance(minutes, int) or isinstance(minutes, list),\
                "minutes must be either list, int or None."
            minutes = Minutes.get_list_minutes(minutes)

        if hours is None:
            hours = list(range(24))
        else:
            hours = Minutes.get_list_hours(hours)

        list_minutes = []
        for hour in hours:
            start_minute = hour*60
            list_minutes.extend([start_minute + minute \
                                 for minute in minutes])
        return list_minutes
    
    @staticmethod
    def combine_days_hours(days: Union[list, int], hours: list)-> list:
        """assumes day 0 and complete list of all minutes in hours
        i.e., assumes combine_hours_minutes has been run already...
         ...hours is a list of minutes!!
        """
        days = Minutes.get_list_days(days)
        
        list_minutes = []
        for day in days:
            start_minute = (day - 1)*1440
            list_minutes.extend([start_minute + minute \
                                 for minute in hours])

        return list_minutes
    
    @staticmethod
    def get_minutes(input: list)-> list:
        """returns a list of minutes based on input"""
        if not isinstance(input, list):
            raise ValueError("Invalid input: input must be a list")
        elif len(input) == 0:
            raise ValueError("Invalid input: no values")
        elif len(input) > 4:
            raise ValueError("Invalid input: too many values")

        input = Minutes.process_input(input)
    
        minutes = Minutes.combine_hours_minutes(input[1], input[2])
        minutes = Minutes.combine_days_hours(input[0], minutes)
        return minutes


    @staticmethod
    def get_days(input: list)-> list:
        """returns a list of days based on input
        
        MADE FOR PROBABILITY: still need to make tests
        """
        if not isinstance(input, list):
            raise ValueError("Invalid input: input must be a list")
        elif len(input) == 0:
            raise ValueError("Invalid input: no values")
        elif len(input) > 2:
            raise ValueError("Invalid input: too many values")

        input = Minutes.process_input(input)
        days = Minutes.get_list_days(input[0])
        return days

    @staticmethod
    def process_input(input: list) -> list:
        """Add None for unspecified values and process months.
        
        TODO: add more testing here
        """
        
        if isinstance(input[0], str) or isinstance(input[0], list):
            if isinstance(input[0], str) or isinstance(input[0][0], str):
                input = Minutes.process_months(input)
        
        
        if len(input) == 1:
            input.append(None)
            input.append(None)
            return input
        elif len(input) == 2:
            input.append(None)
            return input
        elif len(input) ==  3:
            return input
        else:
            raise ValueError("Invalid input: too many values")
        
        
        
    @staticmethod
    def process_months(input: list) -> list:
        """convert input to [D, H, M] format
        
        """
        months = input[0]
        days = input[1]
        input.pop(0)

        if isinstance(months, str):
            month = Minutes.get_month_number(months)
            if isinstance(days, int):
                days = Minutes.get_absolute_day(month, days)
            elif isinstance(days, list):
                for i, day in enumerate(days):
                    days[i] = Minutes.get_absolute_day(month, day)
        elif isinstance(months, list) and len(months) == 2:
            assert len(months) == 2,\
                "Invalid input: month should be a list of 2 strings"
            month1 = Minutes.get_month_number(months[0])
            month2 = Minutes.get_month_number(months[1])
            assert month1 > 0 and month1 < 13,\
                "Invalid input: month1 should be 1-12"
            assert month2 > 0 and month2 < 13,\
                "Invalid input: month2 should be 1-12"
            assert month1 < month2,\
                "Invalid input: month1 should be before month2"
            
            day1 = Minutes.get_absolute_day(month1, days[0])
            day2 = Minutes.get_absolute_day(month2, days[1])

            days = Minutes.get_list_days([day1, day2])

        input[0] = days
        assert len(input) < 4, "Invalid input: too many values"

        return input


        


    @staticmethod
    def get_month_number(month: str) -> int:
        assert isinstance(month, str),\
            "Invalid input: month should be a string"
        valid_strings = [
            ['ja', 'JA', 'january', 'January', 'jan',
             'Jan', 'JANUARY', 'JAN', '01'],
            ['f', 'F', 'february', 'February', 'feb',
             'Feb', 'FEBRUARY', 'FEB', '02'],
            ['march', 'March', 'mar', 'Mar', 'MARCH',
             'MAR', '03'],
            ['a', 'A', 'april', 'April', 'apr', 'Apr',
             'APRIL', 'APR', '04'],
            ['may', 'May', 'MAY', '05'],
            ['june', 'June', 'jun', 'Jun', 'JUNE', 'JUN', '06'],
            ['july', 'July', 'jul', 'Jul', 'JULY', 'JUL', '07'],
            ['a', 'A', 'august', 'August', 'aug', 'Aug',
             'AUGUST', 'AUG', '08'],
            ['s', 'S', 'september', 'September', 'sep',
             'Sep', 'SEPTEMBER', 'SEP', '09'],
            ['o', 'O', 'october', 'October', 'oct',
             'Oct','OCTOBER', 'OCT', '10'],
            ['n', 'N', 'november', 'November', 'nov',
             'Nov', 'NOVEMBER', 'NOV', '11'],
            ['d', 'D', 'december', 'December', 'dec',
             'Dec', 'DECEMBER', 'DEC', '12']
        ]
        for i, month_list in enumerate(valid_strings, start=1):
            if month in month_list:
                return i
        raise ValueError(f"Invalid month: {month}")
    
    @staticmethod
    def days_in_each_month(month: int)-> int:
        """gets number of days in a specific month."""
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return 31
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 28
    
    @staticmethod
    def get_absolute_day(month: int, day: int)-> int:
        """get absolute day number relative to April 1 (day 1).
        
        uses 1-based indexing for days to be consistent
        with how day is specified on a ticket
        (as well as user input).
        """
        if month < 4 or month > 6:
            raise ValueError(("Invalid month: month should be "
                              +"April, May or June"))

        days = 0
        if month > 4:
            for i in range(5, month+1):
                days += Minutes.days_in_each_month(i - 1)
        return days + day
    
    @staticmethod
    def sparse_list_construct(shape=(60, 1440)):
        """constructs a sparse array from a list of minutes"""
        import numpy as np
        from scipy.sparse import lil_matrix
        sparse_list = lil_matrix(shape, dtype=np.int32)
        return sparse_list
    
    @staticmethod
    def sparse_list_add(sparse_list, minutes: list):
        """add minutes to the sparse list"""
        for minute in minutes:
            day = minute // 1440
            minute_of_day = minute % 1440
            sparse_list[day, minute_of_day] = 1
        return sparse_list
    
    @staticmethod
    def get_day_hour_min(minute: int)-> tuple:
        """returns day, hour, minute from minute"""
        day = minute // 1440 + 1
        hour = (minute - (day - 1)*1440) // 60
        min = minute % 60
        return day, hour, min
    
    @staticmethod
    def get_day_min(minute: int)-> tuple:
        """returns day, minute in day from minute"""
        day = minute // 1440 + 1
        min = minute % 1440
        return day, min
    
    @staticmethod
    def day_min(minutes, as_list=False):
        array = np.zeros((len(minutes), 2))
        for i in range(len(minutes)):
            array[i, :] = Minutes.get_day_min(minutes[i])
        if as_list:
            return array.tolist()
        else:
            return array
    
    @staticmethod
    def get_day_hour(minute: int)-> tuple:
        """returns day, hour in day from minute"""
        day = minute // 1440 + 1
        hour = (minute - (day - 1)*1440) // 60
        return day, hour
    
    @staticmethod
    def get_transform(parameters, type="linear"):
        """returns a transformation function based on parameters"""
        if type == "linear":
            d0 = parameters[0]
            d1 = parameters[1]
            m0 = parameters[2]
            m1 = parameters[3]
            def transform(day, min):
                x = (day - d0)/d1
                y = (min - m0)/m1
                return x, y
        else:
            raise ValueError("Invalid type")
        return transform
    
    @staticmethod
    def radius(day, min):
        """cartesian to spherical"""
        return np.sqrt(day**2 + min**2)
    
    @staticmethod
    def angle(day, min):
        """cartesian to spherical"""
        return np.arctan2(min, day)

class Tickets:
    def __init__(self,
                 name="no name",
                 reference="April 1",
                 days=60):
        self.name = name
        self.reference = reference
        self.days = days
        self.cost = 3.0
        self.tickets = []
        self.tickets_sparse = Minutes.sparse_list_construct()

    def status(self):
        print(f"Name: {self.name}")
        print(f"Reference: {self.reference}")
        print(f"Days: {self.days}")
        print(f"Number of tickets: {self.N()}")
        print(f"Cost for all tickets: {self.cost*self.N()}")
        print()

    def N(self, verbose=None) -> int:
        N = len(self.tickets)
        if verbose:
            print(f"Number of tickets: {N}")
        return N

    def show(self):
        fig, ax = plt.subplots()
        for minute in self.tickets:
            day, hour, minute_in_hour = Minutes.get_day_hour_min(minute)
            rect = plt.Rectangle((day - 0.5, hour + minute_in_hour/60 - 0.5/60),
                                 1, 1/60, color='blue')
            ax.add_patch(rect)
        ax.set_xlim(0, self.days)
        ax.set_ylim(0, 24)
        ax.set_xlabel('Day Number (April 1 is Day 0)')
        ax.set_ylabel('Hours')
        ax.set_title('Tickets Overview')
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(6))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        import matplotlib.dates as mdates

        secax = ax.secondary_xaxis('top')
        start_date = datetime.strptime(self.reference, "%B %d")
        date_labels = [start_date + timedelta(days=i) for i in range(self.days)]
        secax.set_xticks(range(0, self.days, 10))
        secax.set_xticklabels([date.strftime("%B %d") for date in date_labels[::10]], rotation=45, ha='left')
        secax.set_xlabel('Date')
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        plt.show()

    def add(self, new: list, verbose=None):
        """add bets (tickets)
        """
        assert isinstance(new, list), "input must be a list"
        new_tickets = Minutes.get_minutes(new)
        self.update(new_tickets, verbose)

    def update(self, new, verbose=None):
        # assert np.all(np.isin(new, [0, 1])), "Invalid input"
        N_old = self.N()
        N_new = len(new)
        _, duplicates, _ = np.intersect1d(new,
                                          self.tickets,
                                          return_indices=True)
        new_net = N_new - len(duplicates)
        assert isinstance(N_old, int), "Non-integer count for N_old"
        assert isinstance(N_new, int), "Non-integer count for N_new"
        assert isinstance(len(duplicates), int), "Non-integer count for duplicates"
        assert isinstance(new_net, int), "Non-integer count for new_net"
        if verbose:
            print(f"Attempting to update ticket list:")
            print(f"  Current: \t\t{N_old}")
            print(f"  New (attempted): \t{N_new}")
            print(f"  Duplicate: \t\t{len(duplicates)}")
            print(f"  Net change: \t\t{new_net}")
            print(f"  New total: \t\t{N_old + new_net}")
            print()
        
        try_update = [i for i in new if i not in self.tickets]
        assert len(try_update) == new_net, "mismatch, number of duplicates!"
        assert len(try_update) == len(set(try_update)), "duplicate values present!"
        
        self.tickets += try_update
        self.tickets.sort()
        assert len(self.tickets) == len(set(self.tickets)), "duplicate values present!"
        assert len(self.tickets) == N_old + new_net, "mismatch, ticket count!"
        assert len(self.tickets) == self.N(), "mismatch, ticket count!"

        self.tickets_sparse = Minutes.sparse_list_add(self.tickets_sparse,
                                                      try_update)
        assert self.tickets_sparse.size == len(self.tickets), "size mismatch, list and lil_matrix"

        if verbose:
            print(f"Update successful! Currently have {self.N()} tickets.\n")

class Models:
    def __init__(self, model_id:str=2, moments_day:tuple=(33, 6.5),
                 moments_min:tuple=(863, 190)):
        self.winnings = 300000
        self.cost = 3
        self.moments_day = moments_day
        self.moments_min = moments_min
        self.distribution_day = stats.norm(self.moments_day[0],
                                           self.moments_day[1])
        self.distribution_min = stats.norm(self.moments_min[0],
                                           self.moments_min[1])
        self.set_ticket_model(model_id=model_id)
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
    
    def set_ticket_model(self, model_id:Union[int, str]=0):
        """set ticket model
        
        0: original exponential equation (fast!)
        1: first model with probability distributions
        """
        if model_id == 0:
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
        elif model_id == 1:
            dict = {
                "references": [28.8327,
                               {"x":[0, 540, 930, 1440],
                                "y":[0., 9.375e-2, 1., 3.125e-2]}
                              ],
                "limits": [0, 50],
                "static_method_name": "ticket_model_1",
                "static_method": self.ticket_model_1
                }
            self.ticket_model_dict = dict
        elif model_id == 2:
            dict = {
                "references": [28.8327,
                               {"mu":863.1425,
                                "std":255.324,
                                "eta":[2,2.2,1.8,300,3,1.6,0.1,0.45]}
                              ],
                "limits": [0, 50],
                "static_method_name": "ticket_model_2",
                "static_method": self.ticket_model_2
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
        

    def plot(self,
             data:Union[np.ndarray, list, int, str, lil_matrix],
             increment:str='min',
             custom_title=None, custom_label=None,
             custom_colors:Union[str, list]=None):
        
        if isinstance(data, int) or isinstance(data, str):
            # data defines which_pickle to use
            if isinstance(data, str):
                data_id = self.pickles.index(data)

            data_id = data
            data = self.pickle_picker('load', data_id)
            titles = ['Historic Probability of Breakup',
                  'Expected Tickets',
                  'Purchased Tickets']
            labels = ['Probability', 'Ticket Count', 'Ticket Count']
            if custom_colors:
                colors = custom_colors
            else:
                colors = ['viridis', 'viridis', 'viridis']
        elif isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, lil_matrix):
            titles = ['Expected Benefit for Selected Tickets']
            labels = ['Expected Benefit (USD)']
            if custom_colors:
                colors = [custom_colors]
            else:
                colors = ['PiYG']
            data_id = 0
        else:
            raise ValueError("data must be an int, str, list, or np.ndarray")
        
        if custom_title:
            titles[data_id] = custom_title
        if custom_label:
            labels[data_id] = custom_label

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
        
        cax = ax.imshow(data.T, aspect='auto',
                        cmap=colors[data_id], origin='lower')
        fig.colorbar(cax, ax=ax, label=labels[data_id])
        
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 24*60)
        ax.set_xlabel('Day Number (April 1 is Day 0)')
        ax.set_ylabel('Minutes of the Day')
        ax.set_title(titles[data_id])
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
                # print(f"Pickle for {pickle_name} is up to date.")

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
                # print(f"Pickle for {pickle_name} is up to date.")
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
    
    @staticmethod
    def ticket_model_1(d: dict):
        """Returns functions; parameters defined by dictionary."""


        reference_day = d["references"][0]
        reference_min = d["references"][1]
        ticket_limits = d["limits"]

        
        
        

        def scale_day(day):

            def poisson(k, mu=reference_day):
                return np.exp(-mu)*mu**k/np.math.factorial(k)
        
            def poisson_pmf_max(mu=reference_day):
                mode = int(np.floor(mu))
                return poisson(mode, mu)
            
            pmf_max = poisson_pmf_max(reference_day)

            pmf_poisson = poisson(day, reference_day)

            return pmf_poisson/pmf_max

        def scale_minute(minute,
                         x=reference_min["x"], y=reference_min["y"]):
            return np.interp(minute, x, y)

        def expected_tickets(day, min):
            day_scale = scale_day(day)
            min_scale = scale_minute(min)
            scale = day_scale*min_scale
            tickets = int(round(
                scale*(ticket_limits[1] - ticket_limits[0]) + ticket_limits[0]
                ))

            return tickets

        return expected_tickets, scale_day, scale_minute
    

    @staticmethod
    def ticket_model_2(d: dict):
        """Returns functions; parameters defined by dictionary."""

        reference_day = d["references"][0]
        reference_min = d["references"][1]
        ticket_limits = d["limits"]

        def get_max_scale_minute(mu=reference_min["mu"],
                                 std=reference_min["std"],
                                 eta=reference_min["eta"]):
            return stats.norm.pdf(mu, mu, std)*eta[1]
        
        max_scale_minute = get_max_scale_minute()


        def scale_day(day):

            def poisson(k, mu=reference_day):
                return np.exp(-mu)*mu**k/np.math.factorial(k)
        
            def poisson_pmf_max(mu=reference_day):
                mode = int(np.floor(mu))
                return poisson(mode, mu)
            
            pmf_max = poisson_pmf_max(reference_day)

            pmf_poisson = poisson(day, reference_day)

            return pmf_poisson/pmf_max

        def scale_minute(minute,
                         mu=reference_min["mu"],
                         std=reference_min["std"],
                         eta=reference_min["eta"]):
            """see colab notebook"""
            
            minute_in_hour = minute % 60  
            
            # bump corrections (0,15,30,and others)
            if minute_in_hour == 0:
                scaling_factor = eta[0]
            elif minute_in_hour==15:
                scaling_factor = eta[1]
            elif minute_in_hour==30:
                scaling_factor = eta[2]
            elif 1<minute<250: # correction for minutes just after midnight
                scaling_factor = eta[3]*1/(minute)
            elif minute==24*60:
                scaling_factor = eta[4]
            elif minute>24*60-5:# correction for minutes just before midnight
                scaling_factor = eta[5]
            else:
                # linearly decreasing correction factor for minutes 1 to 59
                scaling_factor = 1 - eta[6] * (minute_in_hour % 10) / 10 
                
                # linear correction factor that decreases from 1 to 0 over the hour
                minute_scaling_factor = 1 - eta[7]*minute_in_hour / 60.0
                scaling_factor *= minute_scaling_factor 
                    
            
            scaled_value = stats.norm.pdf(minute, mu, std)*scaling_factor

            return scaled_value/max_scale_minute

        def expected_tickets(day, min):
            '''see colab notebook'''
        
            day_scale = scale_day(day)
            min_scale = scale_minute(min)
            scale = day_scale*min_scale

            tickets = int(round(
                scale*(ticket_limits[1] - ticket_limits[0]) + ticket_limits[0]
                ))

            return tickets
        return expected_tickets, scale_day, scale_minute
    


    


    # @staticmethod
    # def data_sparse(data, tickets=None):
    #     """key for plotting stuff"""
    #     return self.map_data_to_day_min(data, tickets=tickets,
    #                                sparse=True)
    
    # @staticmethod
    # def data_zeros(data, tickets=None):
    #     """key for plotting stuff"""
    #     matrix = map_data_to_day_min(data, tickets=tickets)
    #     return matrix
    
    @staticmethod
    def map_data_to_day_min(data, tickets=None,
                            shape=(60, 1440), sparse=False):
        """key for plotting stuff"""

        if tickets is None:
            tickets = np.arange(60*1440).tolist()

        if sparse:
            matrix = lil_matrix(shape)
        else:
            matrix = np.zeros(shape)

        assert len(tickets) == len(data), "tickets and data must be same length"
        count_days_out_of_bounds = 0
        for min, dat in zip(tickets, data):
            day = min // 1440
            minute_of_day = min % 1440
            if day < 0 or day >= shape[0]:
                count_days_out_of_bounds +=1
            else:
                matrix[int(day), int(minute_of_day)] = dat
        if count_days_out_of_bounds > 0:
            print(f"WARNING: map_data_to_day_min: number of "
                +f"days out of bounds: {count_days_out_of_bounds} "
                +f"({count_days_out_of_bounds/len(tickets)*100:.2f}%)")
        return matrix
    @staticmethod
    def reset_prob_pickle():
        """reset probability pickle"""
        import os
        try:
            os.remove('pickles/breakup_prob_hist.pkl')
            print("Pickle reset.")
        except:
            print("Pickle not found.")
        return None


def get_values_in_range(min, max, array1, array2):
    indices = np.where((array1 >= min) & (array1 <= max))
    return array2[indices]

def evaluate_ticket_dist_all(intervals, radius, count, radius_all):

    assert intervals[-1]<=np.max(radius), "intervals must be within max radius"

    d = []
    
    for i in range(len(intervals)-1):
        min = intervals[i]
        max = intervals[i+1]
        val=evaluate_ticket_dist_i(min, max,
                             radius, count, radius_all)
        d.append(val)
        
        
    return d



def evaluate_ticket_dist_i(min, max, radius, minutes, radius_all):
    counts = get_values_in_range(min, max, radius, minutes)
    total = get_values_in_range(min, max, radius_all, radius_all)

    d = RadialDist([min, max], radius, minutes, counts, len(total))

    return d

class RadialDist():
    def __init__(self, range, radius, minutes, counts, total):
        self.range = range
        self.radius = radius
        self.minutes = minutes
        self.counts = counts
        self.N_total = total
        self.N_chosen = len(counts)
        self.N_unchosen = total - self.N_chosen
        self.kde=self.fit_KDE(include_zeros=True)
        self.get_statistics()

    def get_statistics(self):
        d = {}
        mode = stats.mode(self.counts)
        d['mode'] = mode[0]
        d['mode_count'] = mode[1]
        d['mean'] = np.mean(self.counts)
        d['std'] = np.std(self.counts)
        d['min'] = np.min(self.counts)
        d['max'] = np.max(self.counts)
        d['5th'] = np.percentile(self.counts, 5)	
        d['25th'] = np.percentile(self.counts, 25)
        d['median'] = np.median(self.counts)
        d['75th'] = np.percentile(self.counts, 75)
        d['95th'] = np.percentile(self.counts, 95)

        self.stats = d

    def summarize_stats(self):
        print(f"Summary for range {self.range[0]:.2f} to {self.range[1]:.2f}:")
        print(f"  No. min with tickets: {self.N_chosen}")
        print(f"  No. min in range:     {self.N_total}")
        print(f"  No. unchosen tickets: {self.N_unchosen} ({self.N_unchosen/self.N_total*100:.2f}%)")
        print(f"Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value:.3f}")

    def hist(self, include_zeros=False, density=True):
        plt.figure(figsize=(10, 6))

        max = int(np.ceil(np.max(self.counts)))
        bins = np.linspace(0, max, max+1)
        if include_zeros:
            counts = np.append(self.counts, np.zeros(self.N_unchosen))
        else:
            counts = self.counts
        plt.hist(counts, bins=bins, color='b', alpha=0.7, label='Counts',
                density=density)

     
        plt.plot( self.kde, color='r', label='KDE Fit')
        plt.xlabel('Number of Tickets Bought for Range of Minutes')
        if density:
            plt.ylabel('Probability Mass Function for Ticket Count')
        else:
            plt.ylabel('Number of Minutes with Specified Ticket Count')
        plt.title(f'Ticket Distribution for Minutes between {self.range[0]:.2f} and {self.range[1]:.2f} Std Dev of Joint Mean')
        
        return plt.gcf(),self.kde
    def fit_KDE(self, include_zeros=False, bandwidth='scott'):
        if include_zeros:
            counts = np.append(self.counts, np.zeros(self.N_unchosen))
        else:
            counts = self.counts

        kde = stats.gaussian_kde(counts, bw_method=bandwidth)

        values = np.arange(0, np.max(counts) + 1, 1)
        kde_values = kde(values)
        
        #Kde is for continuos distribution, therefore density could be above 1, but we care about discrete, so we need to normalize
        kde_values_normalized = kde_values / np.sum(kde_values)

        return kde_values_normalized


def sample_integer(probabilities, size=1):
    """Get sample given array of integer probabilities.

    Used for sampling ticket numbers for each minute.

    Parameters:
    probabilities (ndarray): An ndarray of probabilities for each discrete integer value.
    size (int): The number of samples to generate.

    Returns:
    ndarray: An ndarray of generated samples.
    """
    values = np.arange(len(probabilities))
    sample = random.choices(values, weights=probabilities, k=size)
    if size == 1:
        return int(sample[0])
    else:
        return np.array(sample, dtype=int)

def sample_ticket(probabilities, cov, N_min,
                  N_max=1000, verbose=False):
    """Sample ticket numbers for each minute."""

    def get_cov_sample(sample, n):
        mean = sample.mean()
        std = sample.std()
        return std/mean/np.sqrt(n)

    sample = sample_integer(probabilities, N_min)

    if np.all(sample == 0):
        if verbose:
            print("All values in the sample are 0.")
        return np.array(sample, dtype=int)
    
    cov_sample = get_cov_sample(sample, N_min)
    max_N_reached = False
    
    while cov_sample > cov:
        N_min += 1
        sample = sample_integer(probabilities, N_min)
        cov_sample = get_cov_sample(sample, N_min)
        if N_min > N_max:
            max_N_reached = True
            break

    if verbose:
        print(f"Sample size: {N_min}, CoV: {cov_sample:0.3f}")
        if max_N_reached:
            print(f"  (maximum sample size reached)")
    
    return np.array(sample, dtype=int)

def plot_hist_and_cdf(sampled_profit, xlim=None, ylim=None):
    fig, ax1 = plt.subplots()

    # Plot histogram
    ax1.hist(sampled_profit, bins=50, edgecolor='k', density=True, alpha=0.6, label='Histogram')
    ax1.set_xlabel('Winnings, thousands of USD')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the cumulative distribution
    ax2 = ax1.twinx()
    sorted_samples = np.sort(sampled_profit)
    cdf = np.arange(1, len(sorted_samples) + 1) / (len(sorted_samples) + 1)
    ax2.plot(sorted_samples, cdf, color='red', label='CDF')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc='upper right')

    # plt.title('Histogram and Cumulative Distribution of Sample Benefits')
    plt.show()

    # Plot exceedance probability
    fig, ax = plt.subplots()
    exceedance_prob = 1 - cdf
    ax.plot(sorted_samples, exceedance_prob, color='blue', label='Exceedance Probability')
    ax.vlines(300, 0, 1, color='grey', linestyle='--', label='300k threshold')
    ax.set_xlabel('Winnings, thousands of USD')
    ax.set_ylabel('Exceedance Probability')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right')
    plt.show()