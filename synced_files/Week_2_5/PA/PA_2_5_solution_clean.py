import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_dict = {}
type(my_dict)

my_dict = {'key': 5}

my_dict['key']

my_dict['array'] = [34, 634, 74, 7345]
my_dict['array'][3]

shell = ['chick']
shell = {'shell': shell}
shell = {'shell': shell}
shell = {'shell': shell}
nest = {'egg': shell}
nest['egg']['shell']['shell']['shell'][0]

new_dict = {'names': ['Gauss', 'Newton', 'Lagrange', 'Euler'],
            'birth year': [1777, 1643, 1736, 1707]}

type(new_dict)

df = pd.DataFrame(new_dict)

type(df)

guess = df.loc[df['birth year'] <= 1700, 'names']
print(guess)

type(guess)

print(type(df.loc[df['birth year'] <= 1700, 'names']))
print(type(df.loc[df['birth year'] <= 1700, 'names'].values))
print('The value in the series is an ndarray with first item:',
      df.loc[df['birth year'] <= 1700, 'names'].values[0])

df.head()

df.describe()

df = pd.read_csv('dams.csv')
df.head()

names_of_earth_dams = df.loc[df['Type'] == 'earth fill', 'Name'].values[:]
print('The earth fill dams are:', names_of_earth_dams)

df_earth = df.loc[df['Type'] == 'earth fill']
df_earth.to_csv('earth_dams.csv', index=False)

