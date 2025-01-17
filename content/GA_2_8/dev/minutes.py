"""creates a class that facilitates conversion and generation of proper month-day-hour-minute specification:

completely define a minute using:
- reference day (day 1)
- hour as a slice object all 1440 hours of the day
- all minutes can be defined in one of the following ways:
  - a collection of slice objects in a list
  - an ndarray of shape (days, 1440) with 1s and 0s
  - list of integers representing consecutive minutes from 0 to 1440*(total days)
  - list of days where each item is either None or a list of integers representing the minutes in that day

Assume that minute 0 of the hour belongs to that hour:
        - 00:00 to 00:59 belongs to hour 0

How to specify:

for any item D, M, H:
- if integer, it is the day, minute, or hour
- if list with length 2, it is a range [start, end] inclusive
- if list with length > 2, it is a list of items
- if list with length 3 has duplicate last 2 items, it is not inclusive

when multiple M or H are specified, they are applied uniformly to the higher order item:

to specify month, first item in the is a string or list containing 2 strings:
- month is a string (several options available, see help(Minutes.get_number_month))
- month can only be specified inclusively; use multiple calls for non-inclusive months
- month must always be accompanied by a day
- [month, D] -> specific day in a single month
- [month, D, H] -> specific hour in a single month
- [month, D, H, M] -> specific minute in a single month
- [month, [D1, D2], H, M] -> inclusive days in a single month
- [[month1, month2], [D1, D2], H, M] -> inclusive days starting in one month, ending in another


Notes:
- for non-inclusive list of 2 items, list one item twice


- [day] -> all minutes of the day
- [day, hour] -> all minutes of the hour
- [day, hour, minute] -> specific minute
- [month, day, hour, minute] -> specific minute

- [[day1, day2]] -> all minutes of the days, inclusive
- [[day1, day2, day3, ...]] -> all minutes of the days
- [[day1, day2, day2]] -> non-inclusive days 1 and 2

- [day, [start_hour, end_hour]] -> all minutes of the hours
- [day, [hour1, hour2, ...]] -> all minutes of the hours
- [day, [hour1, hour2, hour2]] -> non-inclusive hours 1 and 2

- [day, H, M]
- [[day1, day2, ...], H, M] -> as above, H and M applied to all days 
"""
import numpy as np
import matplotlib.pyplot as plt
from array import array
from typing import Union

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