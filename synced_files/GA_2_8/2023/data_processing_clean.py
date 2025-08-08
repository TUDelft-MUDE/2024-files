import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
specific_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)
keep_cols = [0,1,2,7,4]
skip_rows = 0
ice_data = pd.read_excel('./data.xlsx', sheet_name = 'Worksheet', skiprows=skip_rows, usecols=keep_cols)
ice_data['Hour'] = np.floor(ice_data['Hour (24)'])
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['minutes'] = np.abs((ice_data['datetime'] - specific_date).dt.total_seconds() / 60)
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['Year'] = ice_data['datetime'].dt.year
ice_data['Month'] = ice_data['datetime'].dt.month
ice_data['Day'] = ice_data['datetime'].dt.day
ice_data['Hour'] = ice_data['datetime'].dt.hour
ice_data['Minute'] = ice_data['datetime'].dt.minute
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
ice_data['minutes'] = (ice_data['datetime'] - ice_data['ref_date_annual']).dt.total_seconds() / 60
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute
ice_data.to_csv('data.csv', index=False)
ice_data = pd.read_csv('data.csv')
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
ice_data
