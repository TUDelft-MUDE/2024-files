# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
class Person:
    def __init__(self, name):
        self.name = name
    
    def say_hello(self):
        print(f"Hey, I'm {self.name}, nice to meet you!")

class Employee(Person):
    known_salaries = {"janitor": 50000, "professor": 100, "trader": 100000}
    
    def __init__(self, name, job):
        super().__init__(name)
        self.job = job
    
    def salary(self):
        if self.job.lower() in self.known_salaries:
            print(f"My salary is ${self.known_salaries[self.job.lower()]}!")
        else:
            print("I'm not too sure what my salary is :(")

class Friend(Person):
    def say_hello(self):
        print("Yoo, how are you doing? It's lovely to see you again :)")

def greet_person(person):
    person.say_hello()
    print(f"Hey {person.name}!")

# %%
james = Person("James")
james.say_hello()
print()

emma = Employee("Emma", "Janitor")
emma.salary()
emma.say_hello()
print()

philip = Friend("Philip")
greet_person(philip)
print()
greet_person(emma)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(distribution, x_bounds = (-5, 5), function="pdf"):
    X_LOW, X_HIGH = x_bounds

    x_axis = np.linspace(X_LOW, X_HIGH, num=1000)
    
    distribution_function = None
    
    if function == "pdf":
        distribution_function = distribution.pdf
    elif function == "cdf":
        distribution_function = distribution.cdf
    else:
        raise KeyError(f"{function} function not supported")
    
    y_axis = np.vectorize(distribution_function)(x_axis)

    plt.figure(figsize=(15, 5))
    plt.plot(x_axis, y_axis)
    plt.title(function+" of distribution")
    plt.xlabel("x")
    plt.xticks(np.linspace(X_LOW, X_HIGH, num=((X_HIGH - X_LOW) * 2 + 1)))
    plt.ylabel("f(x)")
    plt.show()
    
    return x_axis, y_axis

# %% [markdown]

# %% [markdown]

# %%

distribution = scipy.stats.norm(loc = 1, scale = 3)
x_axis, y_axis = plot_distribution(distribution)

# %% [markdown]

# %% [markdown]

# %%

distribution = scipy.stats.uniform(loc = 1, scale = 3)
x_axis, y_axis = plot_distribution(distribution)

# %% [markdown]

# %% [markdown]

# %%

    

        

class new_distribution(scipy.stats.rv_continuous):
    """ A new piece-wise distribution."""
    
    def _pdf(self, x):
        """ f(x) = 
            0.1      when 0 < x < 3.6
            2(x - 5) when 5 < x < 5.8
        """
        if 0 < x < 3.6:
            return 0.1
        
        if 5 < x < 5.8:
            return 2 * (x - 5)
        
        return 0

# %% [markdown]

# %%
import warnings
warnings.simplefilter("ignore")

# %% [markdown]

# %%

plot_distribution(new_distribution(), x_bounds = (0, 6))
plot_distribution(new_distribution(), x_bounds = (0, 6), function="cdf")
print()

# %% [markdown]

# %% [markdown]

