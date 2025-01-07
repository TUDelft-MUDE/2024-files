import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 14})
h = pd.read_csv(YOUR_CODE_HERE, delimiter=';', header=3)
h.columns=['Date', 'Stage']
h['Date'] = pd.to_datetime(h['Date'], format='%d-%m-%Y %H:%M')
h['Date']
plt.figure(figsize=(10, 6))
plt.plot(h['Date'], h['Stage'],'k.')
plt.xlabel('Date')
plt.ylabel('River Stage [ft]')
plt.grid()
plt.title('River Stage, 2023 Water Year');
idx_max = h.groupby(pd.DatetimeIndex(h['Date']).year)['Stage'].idxmax()
h.loc[idx_max]
plt.figure(figsize=(10, 6))
plt.plot(h['Date'], h['Stage'],'k.', markersize=4)
plt.xlabel('Date')
plt.ylabel('River Stage [ft]')
plt.grid()
plt.title('River Stage, 2023 Water Year')
plt.plot(h[YOUR_CODE_HERE][YOUR_CODE_HERE], h[YOUR_CODE_HERE][YOUR_CODE_HERE], 'y.', markersize=16, markeredgecolor='k')
one_day = 4*24
idx_peaks, _ = find_peaks(h['Stage'], height=1.0, distance=one_day)
print(h.loc[idx_peaks].describe())
plt.figure(figsize=(10, 6))
plt.plot(h['Date'], h['Stage'],'k.', markersize=4)
plt.xlabel('Date')
plt.ylabel('River Stage [ft]')
plt.grid()
plt.title('River Stage, 2023 Water Year')
plt.plot(h['Date'][idx_max], h['Stage'][idx_max],'y.', markersize=16, markeredgecolor='k')
plt.plot(h['Date'][idx_peaks], h['Stage'][idx_peaks],'r.', markersize=16, markeredgecolor='k');
x = 1.24196854
y = 2
string_1 = f'My string with values x = {x:0.3f} and y = {y:0.3f}'
string_2 = 'My string with values x = {:0.3f} and y = {:0.3f}'.format(x, y)
print(string_1)
print(string_2)
print('Second argument is {1:0.3f} and first argument is {0:0.3f}'.format(x, y))
def square(z1, z2):
    'Return tuple with squared values of 2 inputs.'
    return z1*z1, z2*z2
print('Second argument of squared values is {1:0.3f} and first is {0:0.3f}'.format(*square(x,y)))
print(f'{5:7.3f}')
print('^^^^^^^')
print('|||||||')
print('1234567 total width')
print('    123 total decimals')
print(f'{5:7.3f}  <-- same example as above')
print('^^^^^^^')
print('|||||||')
print('1234567 total width')
print(f'{5:1.3f}  <-- total width exceeded; no white space to left of integer')
print('^')
print('|')
print('1 total width (exceeded)')
print('123456789 <-- try printing things with 9 spaces')
print(f'{5:9.3f}')
print(f'{5:9.3e}')
print(f'{5.516654654654:9.9f} <-- oops, f overflowed')
print(f'{5.516654654654:9.9e} <-- e did also')
print(f'{5.516654654654:9.2f} <-- oops, f overflowed')
print(f'{5.516654654654:9.2e} <-- but e did not!')
print(f'{5:9d}')
print(f'{5651651651654:9d} <-- decimal overflow')
string_a = '5'
string_b = '5651651651654'
print(f'{string_a:9s} <-- string left justified')
print(f'{string_b:9s} <-- string overflow')
print('| Column 1 | Column 2 | Column 3 |')
print('| -------- | -------- | -------- |')
print(f'|{0.5:10.3f}|{2.5:10.3f}|{1.5:10f}|')
print(f'|{0.5:10.3f}|{0.5:10.3e}|{2:10d}|')
print(f'|{3:10d}|{0.5:10f}|{0.5:10f}|')
print('| Column 1 | Column 2 | Column 3 |')
print('| -------- | -------- | -------- |')
print(f'|{0.5:<10.3f}|{2.5:<10.3f}|{1.5:<10f}|')
print(f'|{0.5:<10.3f}|{0.5:<10.3e}|{2:<10d}|')
print(f'|{3:<10d}|{0.5:<10f}|{0.5:<10f}|')
print('| Column 1 | Column 2 | Column 3 |')
print('| -------- | -------- | -------- |')
print(f'|{0.5**2:10.3f}|{2.5**2:10.3f}|{1.5**2:10f}|')
print(f'|{0.5**2:10.3f}|{0.5**2:10.3e}|{2**2:10d}|')
print(f'|{3**2:10d}|{0.5**2:10f}|{0.5**2:10f}|')
print('| Column 1 | Column 2 | Column 3 |')
print('| -------- | -------- | -------- |')
print('|{:10.3f}|{:10.3f}|{:10f}|'.format(0.5**2, 2.5**2, 1.5**2))
print('|{:10.3f}|{:10.3e}|{:10d}|'.format(0.5**2, 0.5**2, 2**2))
print('|{:10d}|{:10f}|{:10f}|'.format(3**2, 0.5**2, 0.5**2))
YOUR_CODE_HERE
YOUR_CODE_HERE
