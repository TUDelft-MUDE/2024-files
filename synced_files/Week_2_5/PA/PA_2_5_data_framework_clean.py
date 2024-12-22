# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------
my_dict = {}
type(my_dict)

# ----------------------------------------
my_dict = {'key': 5}

# ----------------------------------------
my_dict['key']

# ----------------------------------------
my_dict['array'] = [34, 634, 74, 7345]
my_dict['array'][3]

# ----------------------------------------
shell = ['chick']
shell = {'shell': shell}
shell = {'shell': shell}
shell = {'shell': shell}
nest = {'egg': shell}
nest['egg']['shell']['shell']['shell'][0]

# ----------------------------------------
new_dict = {'names': ['Gauss', 'Newton', 'Lagrange', 'Euler'],
            'birth year': [1777, 1643, 1736, 1707]}

YOUR_CODE_HERE

# ----------------------------------------
df = pd.DataFrame(new_dict)

YOUR_CODE_HERE

# ----------------------------------------
guess = df.loc[df['birth year'] <= 1700, 'names']
print(guess)

# ----------------------------------------
YOUR_CODE_HERE

# ----------------------------------------
print(type(df.loc[df['birth year'] <= 1700, 'names']))
print(type(df.loc[df['birth year'] <= 1700, 'names'].values))
print('The value in the series is an ndarray with first item:',
      df.loc[df['birth year'] <= 1700, 'names'].values[0])

# ----------------------------------------
df.head()

# ----------------------------------------
df.describe()

# ----------------------------------------
df = YOUR_CODE_HERE
YOUR_CODE_HERE.head()

# ----------------------------------------
names_of_earth_dams = YOUR_CODE_HERE
print('The earth fill dams are:', names_of_earth_dams)

# ----------------------------------------
df_earth = YOUR_CODE_HERE
df_earth.YOUR_CODE_HERE

