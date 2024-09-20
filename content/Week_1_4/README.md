# PA 1.4: Git Me a Table

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3. Due: before Friday, September 27, 2024.*

## Introduction

You can access the assignment with this link: [classroom.github.com/a/s0De54YX](https://classroom.github.com/a/s0De54YX)

This PA consists of 3 parts:

1. Read the first chapter in [Programming for Week 1.4](https://mude.citg.tudelft.nl/2024/book/programming/week_1_4.html) about Version Control
2. Install Git on your computer and set up SSH authentication. as explained in the second chapter
3. Go through the workflow as shown in the third chapter
4. Use VS Code to add a table to the file `my_table.md`, then make a commit and push it to GitHub

The chapters of the textbook contain all the information necessary to complete these steps. Once you have completed the first two steps above, then you can read the instructions below for creating a Markdown table and try it out in `my_table.md`.

**Note: the programming tutorial on Monday of Week 1.4 will cover Git and GitHub, so install Git and attend this session if you anticipate issues with this PA!**

### Passing the PA

You will pass this PA if:
1. You set up Git and SSH on your computer
2. You successfully make a commit on your computer and push the commit to GitHub

You can verify that you passed the assignment by looking for the green circle when you have pushed your commit, it runs automatically just like last week when you uploaded your notebook.

## Formatting Tables in Markdown

This is relatively straightforward, once you learn the proper syntax. Note, however, that the most challenging thing about Markdown tables (and Markdown in general) is that every platform interprets the source code (i.e., the text) differently---in other words, what works in VS Code may not work the same in other IDE's, or even on GitHub. Fortunately this is usually fixed with a little trial-and-error, usually with white space (spaces, new lines, etc) or a quick Google search for some example code.

The following are key notes for constructing a table:
- The essential element is the pipe symbol `|` (the vertical bar), which denotes the column boundaries.
- Each line of code is a new row in the table, which will generate automatically in a Markdown cell as long as you have the same number of pipes in each row.
- The second row is essential, and must contain at least one colon `:` and one hyphen (dash) `-` that tells the Markdown interpreter that you are defining a table (see next comment, and examples).
- You can specify left, center or right justification of text in each column using `:-`, `:-:` or `-:`, respectively. You can also use extra hyphens or whitespace to format the raw text in a readable way.

Below are two example tables to illustrate how it works.

_Table 1: Example table, empty; formatted as source code._

```
|     |     |     |
| :-: | :-: | :-: |
|     |     |     |
```

_Table 2: Example with left, center, right justification in columns._

| Column 1 | Column 2 | Column 3 |
| :- | :-: | -: |
| Left Justified | Center | Right |
| a | b | c |
| 1 | 2 | 3 |

**Note about viewing the examples:** _if you are viewing this file on GitHub, the tables above probably display properly, and you will have to click the "raw" or "view source" button to see the source code. If you are viewing this file in a text editor, you will have to use a tool to view the "marked up" version. In VS Code use the "markdown all-in-one" extension (hit `CTRL+SHIFT+V` once it is installed). This is also useful for this assignment to verify that the table formats properly._

### PA 1.4: Try it Yourself!

Now you know everything that is needed to create your own Markdown table, and you can do this in the file `my_table.md` to complete the assignment.

Edit the file in VS Code, then use the viewing extension to visualize it. You can fill it with any contents you like, and make it big or small. Once you are satisfied with your table, do the following:
- make a commit
- push the commit to your repository on GitHub
- check the green checkbox to make sure you have passed the assignment

Congratulations, you're done!

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.