import numpy as np
import matplotlib.pyplot as plt
from minutes import *
from datetime import datetime, timedelta
from scipy.sparse import lil_matrix

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

    




