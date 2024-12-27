<userStyle>Normal</userStyle>

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
---

# PA 2.5: Data Framework

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.5. Due: complete this PA prior to class on Friday, Dec 13, 2024.*


## Overview of Assignment

This assignment quickly introduces you to the package `pandas`. We only use a few small features here, to help you get familiar with it before using it more in the coming weeks. The primary purpose is to easily load data from csv files and quickly process the contents. This is accomplished with a new data type unique to pandas: a `DataFrame`. It also makes it very easy to export data to a `*.csv` file.

If you want to learn more about pandas after finishing this assignment, the [Getting Started page](https://pandas.pydata.org/docs/getting_started/index.html) is a great resource.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Part 1: Introduction to pandas

Pandas dataframes are considered by some to be difficult to use. For example, here is a line of code from one of our notebooks this week. Can you understand what it is doing?
```
net_data.loc[net_data['capacity'] <= 0, 'capacity'] = 0
```

One of the reasons for this is that the primary pandas data type, a `DataFrame` object, uses a dictionary-like syntax to access and store elements. For example, remember that a dictionary is defined using curly braces. 

```python
my_dict = {}
type(my_dict)
```

Also remember that you can add items as a key-value pair:

```python
my_dict = {'key': 5}
```

The item `key` was added with value 5. We can access it like this:

```python
my_dict['key']
```

This is useful beceause if we have something like a list as the value, we can simply add the index the the end of the call to the dictionary. For example:

```python
my_dict['array'] = [34, 634, 74, 7345]
my_dict['array'][3]
```

And now that you see the "double brackets" above, i.e., `[ ][ ]`, you can see where the notation starts to get a little more complicated. Here's a fun nested example:

```python
shell = ['chick']
shell = {'shell': shell}
shell = {'shell': shell}
shell = {'shell': shell}
nest = {'egg': shell}
nest['egg']['shell']['shell']['shell'][0]
```

Don't worry about that too much...as long as you keep dictionaries and their syntax in mind, it becomes easier to "read" the complicated pandas syntax.

Now let's go through a few simple tasks that will illustrate what a `DataFrame` is (when constructed from a dictionary), and some of its fundamental methods and characteristics.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1:</b>   
    
Run the cell below and check what kind of object was created using the method <code>type</code>.
</p>
</div>

```python
new_dict = {'names': ['Gauss', 'Newton', 'Lagrange', 'Euler'],
            'birth year': [1777, 1643, 1736, 1707]}

YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2:</b>   
    
Run the cell below and check what kind of object was created using the method <code>type</code>.
</p>
</div>

```python
df = pd.DataFrame(new_dict)

YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3:</b>   
    
Read the code below and try to predict what the answer should be before you run it and view the output. Then run the cell, confirm your guess and in the second cell check what kind of object was created using the method <code>type</code>.
</p>
</div>

```python
guess = df.loc[df['birth year'] <= 1700, 'names']
print(guess)
```

```python
YOUR_CODE_HERE
```

Note that this is a `Series` data type, which is part of the pandas package (you can read about it [here](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)). If you need to use the value that is stored in the series, you can use the attribute `values` as if it were an object with the same `type` as the data in the `Series`; the example below shows that the `names` in the `DataFrame` is a `Series` where the data has type `ndarray`.

```python
print(type(df.loc[df['birth year'] <= 1700, 'names']))
print(type(df.loc[df['birth year'] <= 1700, 'names'].values))
print('The value in the series is an ndarray with first item:',
      df.loc[df['birth year'] <= 1700, 'names'].values[0])
```

Another useful feature of pandas is to be able to quickly look at the contents of the data frame. You can quickly see which columns are present:

```python
df.head()
```

You can also get summary information easily:

```python
df.describe()
```

Finally, it is also very easy to read and write dataframes to a `*.csv` file, which you can do using the following commands (_you will apply this in the tasks below_):
```
df = pd.read_csv('dams.csv')
```
To write, the method is similar; the keyword argument `index=False` avoids adding a numbered index as an extra column in the csv:
```
df.to_csv('dams.csv', index=False)
```


## Task 2: Evaluate and process the data

For this assignment we will use a small files `dams.csv` file that is available in the repository for this PA.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.1:</b>
    
Import the dataset as a DataFrame, then explore it and learn about its contents (use the methods presented above; you can also look inside the csv file).
</p>
</div>

```python
df = YOUR_CODE_HERE
YOUR_CODE_HERE.head()
```

<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

We can see that this dataset has some information about dams, including the name, year constructed, volume and height. They look pretty big! It's actually the largest 5 dams by either volume or height (10 dams total), listed on Wikipedia page <a href="https://en.wikipedia.org/wiki/List_of_largest_dams" target="_blank">here</a>.

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.2:</b>
    
Using the example above, find the dams in the <code>DataFrame</code> that are of type <code>earth fill</code>.</code>
</p>
</div>

```python
names_of_earth_dams = YOUR_CODE_HERE
print('The earth fill dams are:', names_of_earth_dams)
```

_Hint: the answer should be:_ `['Fort Peck' 'Nurek' 'Kolnbrein' 'WAC Bennett']`


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.3:</b>
    
Create a new dataframe that only includes the earth fill dams. Save it as a new csv file called <code>earth_dams.csv</code>.
</p>
</div>

_Hint: you only need to remove a small thing from the code for your answer to the task above)._

```python
df_earth = YOUR_CODE_HERE
df_earth.YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.4:</b>
    
Check the contents of the new csv file to make sure you created it correctly.
</p>
</div>


**End of notebook.**
<h2 style="height: 60px">
</h2>
<h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    
</h3>
<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
