
import numpy as np
import matplotlib.pyplot as plt

team = ['green', 'red', 'blue']
score = [5, 9, 7]

N_teams = len(team)
print('The team names are:')
for i in range(N_teams):
    print(f'{team[i]}')

iter(team)

iter(np.array([1, 5, 67]))

iter(5)

letters = 'abcde'
letter_iterator = iter(letters)

next(letter_iterator)

for i in letters:
    print(i)

print(type(range(5)))
print(range(5))

for i in range(5):
    print(i)

thing_1 = 'roberts_string'
thing_2 = [2, 3, 36, 3., 1., 's', 7, '3']

test_1 = enumerate(thing_1)
print(f'We created: {test_1}')
print(next(test_1), next(test_1), next(test_1))

test_2 = zip(thing_1, thing_2)
print(f'We created: {test_2}')
print(next(test_2), next(test_2), next(test_2))

print('First, enumerate:')
for i, j in enumerate(thing_1):
    print(i, j)
print('\nThen, zip:')    
for i, j in zip(thing_1, thing_2):
    print(i, j)

team = ['green', 'red', 'blue']
score = [5, 9, 7]

for YOUR_CODE_WITH_enumerate_HERE:
    print(f'Team {} has {} points.')

team = ['green', 'red', 'blue']
score = [5, 9, 7]

for YOUR_CODE_WITH_zip_HERE:
    print(f'Team {} has {} points.')

print(6 % 5)
print(5 % 6)
print(1 % 10)
print(5 % 5)
print(0 % 5)

for i in range(100):
    if i%25 == 0:
        print(i)

value = [2, 7, 5, 1, 8]
index = [0, 2, 2, 6, 4]
plt.plot(index, value, 'o')
plt.stem(index, value);

year = [2003, 2011, 2013,
        2006, 2022,
        2017, 2019, 2001, 2010, 2015,
        2014, 2022, 2016,
        2000, 2007, 2012, 2005, 2004]

magn = [8.3, 9.1, 8.3,
        8.3, 7.6,
        8.2, 8.0, 8.4, 8.8, 8.3,
        8.2, 7.6, 7.9,
        8.0, 8.4, 8.6, 8.6, 9.1]

site = ['Japan, Hokkaidō',
        'Japan, Honshu',
        'Russia, Sea of Okhotsk',
        'Russia, Kuril Islands',
        'Mexico, Michoacán',
        'Mexico, Chiapas',
        'Peru, Loreto',
        'Peru, Arequipa',
        'Chile, Concepción',
        'Chile, Coquimbo',
        'Chile, Iquique',
        'Papua New Guinea, Morobe',
        'Papua New Guinea, New Ireland',
        'Papua New Guinea',
        'Indonesia, Sumatra',
        'Indonesia, Indian Ocean',x
        'Indonesia, Simeulue',
        'Indonesia, Sumatra']

YOUR_CODE_HERE

plt.savefig('my_earthquake.svg') # Don't remove this line

