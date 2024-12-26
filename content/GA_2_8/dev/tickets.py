import numpy as np
import matplotlib.pyplot as plt

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

    def N(self, verbose=None) -> int:
        N = np.sum(self.tickets)
        assert N % 1 == 0, "Non-integer ticket count"
        N = int(N)
        if verbose:
            print(f"Number of tickets: {N}")
        return N
    
    def show(self):
        compressed_tickets = self.tickets.reshape(self.days, 24, 60).sum(axis=2).T
        plt.matshow(compressed_tickets, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        plt.xlabel('Days')
        plt.ylabel('Hours')
        plt.title('Tickets Overview')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(6))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        plt.show()

    def add(self, new: list, inclusive=True, verbose=None):
        """add bets (tickets)
        List

        Assume that minute 0 of the hour belongs to that hour:
        - 00:00 to 00:59 belongs to hour 0
        """
        assert isinstance(new, list), "Invalid input"
        if len(new)==1:
            if verbose:
                print(f"Add tickets for all of day {new[0]}")
            new_tickets = self.empty
            new_tickets[new[0], :] = 1
        elif len(new) == 2:
            if isinstance(new[1], list) and len(new[1]) == 2:
                if verbose:
                    print(f"Add tickets for day {new[0]} and hours {new[1]} with inclusive={inclusive}")
                new_tickets = self.empty
                start_hour, end_hour = new[1]
                if inclusive:
                    for hour in range(start_hour, end_hour + 1):
                        minutes = slice(hour * 60, (hour + 1) * 60)
                        new_tickets[new[0], minutes] = 1
                else:
                    for hour in new[1]:
                        minutes = slice(hour * 60, (hour + 1) * 60)
                        new_tickets[new[0], minutes] = 1
            else:
                if verbose:
                    print(f"Add tickets for day {new[0]} and hour {new[1]}")
                new_tickets = self.empty
                minutes = slice(new[1] * 60, (new[1] + 1) * 60)
                new_tickets[new[0], minutes] = 1
        
        else:
            print("Unknown ticket specification, try again.")
            return
        self.update(new_tickets, verbose)

    def update(self, new, verbose=None):
        assert np.all(np.isin(new, [0, 1])), "Invalid input"
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
        assert np.all(np.isin(try_update, [0, 1])), "Invalid ticket count"

        self.tickets += new_net
        if verbose:
            print(f"Update successful")
            self.N(verbose)

    




