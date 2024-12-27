import numpy as np
import matplotlib.pyplot as plt
from minutes import *
from datetime import datetime, timedelta
class Tickets:
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

    def status(self):
        print(f"Name: {self.name}")
        print(f"Reference: {self.reference}")
        print(f"Days: {self.days}")
        print(f"Number of tickets: {self.N()}")
        print(f"Cost for all tickets: {self.cost*self.N()}")
        print()

    def N(self, verbose=None) -> int:
        N = np.sum(self.tickets)
        assert N % 1 == 0, "Non-integer ticket count"
        N = int(N)
        if verbose:
            print(f"Number of tickets: {N}")
        return N

    def show(self):
        fig, ax = plt.subplots()
        for day in range(self.days):
            for minute in range(1440):
                if self.tickets[day, minute] == 1:
                    hour = minute // 60
                    minute_in_hour = minute % 60
                    rect = plt.Rectangle((day, hour + minute_in_hour / 60), 1, 1 / 60, color='blue')
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
        List

        Assume that minute 0 of the hour belongs to that hour:
        - 00:00 to 00:59 belongs to hour 0
        """
        assert isinstance(new, list), "Invalid input"
        minutes = Minutes.get_minutes(new)
        new_tickets = Minutes.construct_sparse_array(minutes)
        self.update(new_tickets, verbose)

    def update(self, new, verbose=None):
        # assert np.all(np.isin(new, [0, 1])), "Invalid input"
        N_old = self.N()
        N_new = np.sum(new)
        duplicates = (self.tickets + new) > 1
        new_net = (self.tickets + new) == 1
        assert N_old%1 == 0, "Non-integer count for N_old"
        assert N_new%1 == 0, "Non-integer count for N_new"
        assert np.sum(duplicates)%1 == 0, "Non-integer count for duplicates"
        assert np.sum(new_net)%1 == 0, "Non-integer count for new_net"
        if verbose:
            print(f"Currently have: {int(N_old)}")
            print(f"Attempting to add: {int(N_new)}")
            print(f"Duplicates: {int(np.sum(duplicates))}")
            print(f"Net change: {int(np.sum(new_net))}")
            print(f"New total: {int(N_old + np.sum(new_net))}")
        
        try_update = self.tickets + new_net

        assert duplicates.shape == self.tickets.shape, "Shape mismatch: duplicates"
        assert new_net.shape == self.tickets.shape, "Shape mismatch: net"
        assert new.shape == self.tickets.shape, "Shape mismatch: new"
        assert np.sum(try_update) == N_old + np.sum(new_net), "Ticket count mismatch"
        # assert np.all(np.isin(try_update, [0, 1])), "Invalid ticket count"

        self.tickets += new_net
        if verbose:
            print(f"Update successful")
            self.N(verbose)

    




