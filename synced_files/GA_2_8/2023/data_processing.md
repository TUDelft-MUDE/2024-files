<userStyle>Normal</userStyle>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
```

```python
specific_date = datetime(year=2024, month=4, day=1, hour=0, minute=0)

# Define columns to keep
keep_cols = [0,1,2,7,4]

# Define rows to skip
skip_rows = 0

# Import data 
ice_data = pd.read_excel('./data.xlsx', sheet_name = 'Worksheet', skiprows=skip_rows, usecols=keep_cols)

# Create a column with only hours in 24h format 
ice_data['Hour'] = np.floor(ice_data['Hour (24)'])

# Creates new column in datetime format
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Calculate the difference in minutes
ice_data['minutes'] = np.abs((ice_data['datetime'] - specific_date).dt.total_seconds() / 60)

#calculate minute of the day
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute

# Make sure all data is in the right format
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['Year'] = ice_data['datetime'].dt.year
ice_data['Month'] = ice_data['datetime'].dt.month
ice_data['Day'] = ice_data['datetime'].dt.day
ice_data['Hour'] = ice_data['datetime'].dt.hour
ice_data['Minute'] = ice_data['datetime'].dt.minute

# Create a new column for the specific date (April 1st at 00:00) of the same year
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')

# Calculate the difference in minutes
ice_data['minutes'] = (ice_data['datetime'] - ice_data['ref_date_annual']).dt.total_seconds() / 60

#calculate minute of the day
ice_data['minutes_in_day'] = ice_data['datetime'].dt.hour * 60 + ice_data['datetime'].dt.minute
```

```python
ice_data.to_csv('data.csv', index=False)
ice_data = pd.read_csv('data.csv')
ice_data['datetime'] = pd.to_datetime(ice_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
ice_data['ref_date_annual'] = pd.to_datetime(ice_data['Year'].astype(str) + '-04-01')
```

```python
ice_data
```

```python

```

<!-- #region -->
**End of notebook.**

<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
  <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
  </div>
  <div style="font-size: 75%; margin-top: 10px; text-align: right;">
    By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
    &copy; 2024 TU Delft. 
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
    <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
