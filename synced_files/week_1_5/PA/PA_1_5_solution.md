---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
---

# PA 1.5: A Few Useful Tricks

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px; height: auto; margin: 0" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px; height: auto; margin: 0" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.5. Due: before Friday, Oct 4, 2024.*


The purpose of this notebook is to introduce a few useful Python and Markdown topics:

1. assert statements
2. list comprehension
3. plt.bar()
4. including a figure in a Markdown document

<!-- #region -->
## Topic 1: Assert Statements

Assert statements are like small little "fact-checkers" that you can insert into your code to make sure it is doing what you think it should do. For example, if a matrix should be 4x6 you can use an assert statement to check this; if the matrix is not the right size, and error will occur. Although your first thought may be "why would I want to make my code break with an error?!?!" it turns out this is a very useful debugging and testing tool (something we will cover in a later PA as well). The reason it is useful is that it causes your code to break for a _very specific_ reason, so that you can identify problems and fix them efficiently and confidently. Sounds useful, right?! Luckily assert statements are also quite simple:

There are two cases of this statement that we will use:

Case 1:
```
assert <logical_argument>
```

Case 2:
```
assert <logical_argument>, 'my error message'
```

The best way to illustrate the use of this tool is by example. But first, some important information about the `<logical_argument>`.

### Background Information: Logical Statements

Another key aspect for using this tool effectively is to be aware of what a **logical statement** is and how they can be specified in Python.

A **logical statement** is simply a statement that can be evaluated as **true or false**. This is sometimes referred to as a **binary**, or, in computing in particular, a **Boolean**. Here are some examples of logical statements, formulated as questions, that have a binary/Boolean results:
- Is it raining?
- Is my computer on?
- Were you in class yesterday?

Here are some examples of questions that cannot be evaluated as a logical statement:
- How long will it rain for?
- When will my computer battery reach 0%?
- How many lectures have you skipped this week?



Each of these examples 

Python expression that checks whether or not something is 


| Name of Logical Statement | Python Syntax |
| :----: | :----: |
| Equals | `a == b` |
| Not Equals | `a != b` |
| Less than | `a < b` |
| Less than or equal to | `a <= b` |
| Greater than | `a > b` |
| Greater than or equal to | `a >= b` |

<!-- #endregion -->

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1:</b>   
Execute the cell below and observe the error. Note that it very specifically is an <code>AssertionError</code>.

See if you can fix the code to prevent the error from occurring.
</p>
</div>

```python
# x = 0
# assert x == 1

# SOLUTION
x = 1
assert x == 1
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2:</b>   
Now try fixing the cell below by 1) adding your own error message (see Case 2 above), and 2) forcing the assert statement to fail. Confirm that you can see error message in the error report.
</p>
</div>

```python
# y = 0
# assert y != 1, YOUR_MESSAGE_HERE

# SOLUTION
y = 0
assert y != 1, "y should not be 1 but it is"
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3:</b>   
Explore assert statements by writing your own. Experiment with different data types (e.g., strings, floats, arrays, lists) and use different logical statements from the list above.
</p>
</div>

```python
# YOUR_CODE_HERE
```

### Summary of Asserts

You are now an expert on `assert` statements. Remember these key points:

- an assert statement answers a true/false question; it will will cause an error if false and do nothing if true
- the syntax is `assert <logical_statement>`
- you can easily customize the error message with the syntax `assert <logical_argument>, 'my error message'`
- the `<logical_statement>` must be a Boolean result


## Part 2: List Comprehension

List and dictionary *comprehensions* are elegant constructs in Python that allow you to manipulate Python objects in a very compact and efficient way. You can think of them as writing a `for` loop in a single line. Here is the syntax:

```
[<DO_SOMETHING> for <ITEM> in <ITERABLE>]
```

Note the following key elements:
- the list comprehension is enclosed in list brackets: `[ ... ]`
- `<DO_SOMETHING>` is any Python expression (for example, `print()` or `x**2`) 
- `<ITEM>` is generally a (temporary) variable that is used in the expression `<DO_SOMETHING>` and represents all of the "items" in `<ITERABLE>`
- `<ITERABLE>` is an _iterable_ object. Don't worry about what this is, exactly (we will study it more later).

For our purposes, it is enough to consider the following _iterables_:
- lists (e.g., `[1, 2, 3]`)
- ndarrays (Numpy)
- `range()`
- dictionaries

As with assert statements, the best way to illustrate this is by example.


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.1:</b>   
Read the cell below then execute it to see an example of a "normal" for loop that creates a list of squares from 0 to 9.
</p>
</div>

```python
squares = []
for i in range(10):
    squares.append(i**2)

print(squares)
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.2:</b>   
Read the cell below then execute it to see an example of a list comprehension that does the same thing. It's much more compact, right?!
</p>
</div>

```python
squares = [i ** 2 for i in range(10)]

print(squares)
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.3:</b>   
Read the cell below then execute it to see an example of a "normal" for loop that creates a dictionary that maps numbers to their squares.
</p>
</div>

```python
squares_dict = {}
for i in range(10):
    squares_dict[i] = i ** 2

print(squares_dict)
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.4:</b>   
Read the cell below then execute it to see an example of a list comprehension that does the same thing. It's much more compact, right?!
</p>
</div>

```python
squares_dict = {i: i ** 2 for i in range(10)}

print(squares_dict)
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.5:</b>   
Now try it yourself! Create a new list from from the one defined below that has values that are half that of the original.

<em>Note the use of asserts to make sure your answer is correct!</em>
</p>
</div>

```python
# my_list = [1, 2, 3, 4, 5]
# new_list = []
# print(new_list)
# assert new_list == [0.5, 1.0, 1.5, 2.0, 2.5], "new_list values are not half of my_list!" 

# SOLUTION
my_list = [1, 2, 3, 4, 5]
new_list = [x/2 for x in my_list]
assert new_list == [0.5, 1.0, 1.5, 2.0, 2.5], "new_list values are not half of my_list!" 
```

### Summary

There are several reasons why you should use list comprehension, hopefully you can recognize them from the examples and tasks above:

- Readability: Comprehensions often turn multiple lines of code into a single, readable line, making the code easier to understand at a glance.
- Efficiency: They are generally faster because they are optimized in the Python interpreter.
- Simplicity: Reduces the need for loop control variables and indexing, making the code simpler.

Sometimes the hardest thing to remember is the order and syntax. The following list comprehension uses obvious variable names to illustrate it (assuming you have an object with "stuff" in it, for example, `objects = [1, 2, 3]`); if you can remember this, you can remember list comprehensions!

```
[print(object) for object in objects]
```


## The `plt.bar()` Method

At this point we have created many figures in MUDE assignments using a method from the Matplotlib plotting library: `plt.plot()`. This is our "bread and butter" plot because it is so easy to plot lines, data, scatter plots, etc. However, there are _many_ more types of plots available in Matplotlib. Today we will try `bar()`, which, as you can imagine, creates bar plots.

First take a look at the documentation and see if you can figure out how it works.


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.1:</b>   
Run the cell below and read the docstring for the method. Can you determine the minimum type of inputs required, and what they will do?
</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
help(plt.bar)
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.2:</b>   
That's right, it plots a bar chart where the first argument is the x coordinate of the bar and the second argument is the height. Fill in the empty lists below to create a bar plot with any values you like.
</p>
</div>

```python
# plt.bar([], [])

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6])
```

Pretty easy, right? Let's try to do one more thing with this - suppose we don't like that the _center_ of the bar is over the value we enter. It's easy to change this using a _keyword argument_; these are the input arguments to the function that have the equals sign (e.g., `function(keyword_arg=<xxx>)`). These are optional arguments; they are generally not needed, but can be specified, along with a value, to change the default behavior of the function. For our purposes this week, we will want to change _two_ keyword arguments:

1. `width`
2. `align`

Fortunately the `help` function printed out the docstring for `bar()`, which contains all the information you need to figure out what these keyword arguments do and how to use them.


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.3:</b>   
Set the keyword arguments below to make the bars fill up the entire space between each bar (no white space) and to force the <b>left</b> side of the bar to align with the value specified.

Note the addition of keyword argument <code>edgecolor</code> to make it easier to see the edges of the bar. 
</p>
</div>

```python
# plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
#         width=YOUR_CODE_HERE,
#         align=YOUR_CODE_HERE,
#         edgecolor='black')

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=1,
        align='edge',
        edgecolor='black')
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.4:</b>   
Now set the keyword arguments below to make the bars fill up the entire space between each bar (no white space) and to force the <b>right</b> side of the bar to align with the value specified.
</p>
</div>

```python
# plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
#         width=YOUR_CODE_HERE,
#         align=YOUR_CODE_HERE,
#         edgecolor='black')

# SOLUTION
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6],
        width=-1,
        align='edge',
        edgecolor='black')
```

## Topic 4: Exporting a Figure and Including it in a Markdown Notebook

Now that we can create a wider variety of figures, we should be able to include them in our Reports for communicating the results of our analyses. Here we show you a very simple way to save a figure generated in your notebook, then use Markdown to visualize the figure. Once a figure is made it is easy, use this syntax:

```
![<an arbitrary label for my figure>](<relative path to my figure>)
```

The label is simply a name that will appear in case the figure fails to load. It can also be read by a website-reading app (for example, then a blind person could understand what the content of the figure may be). Here is an example for what this could look like in practice:

```
![bar chart of dummy data](./my_bar_chart.svg)
```

This is very easy, so once again we lead by example! However, first a couple notes about filepaths.

### File Paths

A _file path_ is like an address to a file. There are generally two types, _absolute_ and _relative._ Most of our activities focus on working in a  _working directory,_ so we will focus almost entirely on relative paths. The general format is like this:

```
./subdirectory_1/subdir_2/filename.ext
```

where:
- the dot `.` indicates one should use the current directory of the file (or the CLI) as the current location (the `.` is like saying " start _here_")
- forward slashes `/` separate subdirectories
- the last two words are the file name and extension. For example, common image extensions are `.jpg`, `.png` or `.svg`
- in this example, the image file is stored inside a folder called `subdir_2` which is inside a folder called `subdirectory_1` which is in our working directory.

As a general rule, **always use forward slashes whenever possible.** Although backward slashes are the default and must be used at times on Windows, they don't work on Mac or Linux systems. This causes problems when sharing code with others running these systems (or when we check your assignments on GitHub!!!). Remember that we try to do things in a way that allows easy collaboration: using approaches that are agnostic of the operating system (i.e., works on all platforms). This is hard to guarantee in practice, but consistently using forward slashes will get us close!

### Try it out yourself

We will try this out, but first we need to create a figure! The code below is a quick way of saving a Matplotlib figure as an svg file.


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.1:</b>   
Run the cell below to create the svg file. Confirm that it is created successfully by examining your working directory.
</p>
</div>

```python
fig, ax = plt.subplots(1,1,figsize = (8,6))
plt.bar([1, 2, 3, 4],[0.2, 0.5, 0.1, 0.6])
fig.savefig('my_figure.svg')

```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.2:</b>   
Now that the image is created, use the Markdown cell below to display it in this notebook.
</p>
</div>


Use this Markdown cell to try visualizing the figure we just saved using Markdown!

![a figure]()


**SOLUTION**

```
![a figure](./my_figure.svg)
````

![a figure](./my_figure.svg)


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.3:</b>   
Test your understanding of relative file paths by moving the csv file to a subdirectory with name <code>figure</code> and getting the following Markdown cell to display the figure.

<pre>
<code>
![a figure](./figures/my_figure.svg)
</code>
</pre>

</p>
</div>


** MAKE SURE A FIGURE APPEARS HERE BY MODIFYING THIS MARKDOWN CELL**

![a figure](./test/my_figure.svg)


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4.4:</b>   
Now add the figure to an actual markdown while, specifically <code>myfigure.md</code>. Then visualize to confirm you did it properly (remember to use `CTRL+SHIFT+V` in VSC, the [Markdown All-in-one extension](https://mude.citg.tudelft.nl/2024/book/external/learn-programming/book/install/ide/vsc/extensions.html#markdown-all-in-one))

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
