<userStyle>Normal</userStyle>

# WS 2.8: Dirty Water

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.8. Wednesday January 15, 2024.*


## Introduction

In this exercise we apply a few simple concepts from the textbook to derive a safety standard to regulate spills in a chemical factory. You are asked to consider an economic and societal standard, then make a recommendation to the city council.

## Case Study

A city with population 10,000 uses an aquifer for its water supply, as illustrated in the figure. The city owns a factory in the region that manufactures hazardous chemicals, and recently a chemical spill occurred that resulted in 10 residents getting sick and total damages of  €7,000M*. The city is going to enforce stricter regulations on the factory, and _you have been hired to advise the city council on the maximum allowable probability of a spill occurring (per year)_. You will make a recommendation based on the more stringent criteria between economic and societal risk limits.

![](./images\sketch.png)

Experts have been consulted and it appears under the current plan the probability of a spill is 1/100 per year. The city council is considering two strategies to upgrade the spill prevention system. A small upgrade would cost €25M and can reduce spill probability by a factor 10; a large upgrade with investment costs of €50M would reduce the probability by factor 100.

The city has also considered the regulations in a nearby region which uses a maximum allowable probability of 1 person getting sick as $p_f=0.01$. The city agrees with this, however, they are very much _risk averse_ (that's a hint!), regarding spills with more significant consequences.

_*M = million, so 7,000M is 7e9, or 7 billion. All costs in this exercise are expressed in units €M._

```python
import numpy as np
import matplotlib.pyplot as plt
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 1:</b>  

What is best strategy in terms of total cost of the system?
</p>
</div>


_Your answer here._


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2:</b>  
Assuming that the number of people in a future spill would also be 10 people, what is the maximum allowable spill probability based on societal risk standards?

Make a plot of the societal risk limit and add a point for the case of the city factory.
</p>
</div>

```python
C = YOUR_CODE_HERE
alpha = YOUR_CODE_HERE

pf_societal = YOUR_CODE_HERE
print(pf_societal)

N_values = YOUR_CODE_HERE
limit_line = YOUR_CODE_HERE

fig, ax = plt.subplots(figsize=(8, 6))
YOUR_CODE_HERE
YOUR_CODE_HERE
ax.set_title('YOUR_CODE_HERE')
ax.set_xlabel('YOUR_CODE_HERE')
ax.set_ylabel('YOUR_CODE_HERE')
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
ax.grid(True)
plt.show()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3:</b>  

Provide your advice to the polder’s authorities for the safety standard based on the outcomes of the economic analysis (Task 1) and societal risk (Task 2).

</p>
</div>


_Your answer here._


## Additional Risk Analysis

It turns out that since the spill occurred an evaluation was completed by a third party company, where they identified the following scenarios, along with estimated a probability of each event occurring. The city would like you to see if it would conflict with your safety recommendations. The results of the risk analysis are provided as the probability associated with a specific number of people getting sick; that is:

$$
p_i = P[n_{i-1} < N \leq n_i]
$$

For $i=1:N_s$, where $N_s$ is the number of scenarios; specific values of the consequence, $n_i$, of each scenario $i$ are provided in the Python code cell below, along with the associated probability. Note that the intervals of $N$ are used to simplify the analysis, rather than provide a specific value for every possible number of people getting sick.




<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4:</b>  

Create the FN curve for the factory and compare it with the societal risk limit you determined above.
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>
<b>Hint:</b> you may find it useful to use <code>np.cumsum</code> as well as Matplotlib's <code>step</code> function (in particular, consider whether to use <code>pre</code>, <code>post</code> or <code>mid</code> for keyword argument <code>where</code>).
</p></div>

```python
n_and_p = np.array([[1,   0.099],
                    [2,   7e-4],
                    [5,   2e-4],
                    [10,  9e-5],
                    [20,  5e-6],
                    [60,  3e-6],
                    [80,  6e-7],
                    [100,  4e-7],
                    [150, 8e-8]])

N_plot = n_and_p[:, 0]

p_cumulative = YOUR_CODE_HERE

fig, ax = plt.subplots(figsize=(8, 6))
YOUR_CODE_HERE
YOUR_CODE_HERE
YOUR_CODE_HERE
ax.set_title('YOUR_CODE_HERE')
ax.set_xlabel('YOUR_CODE_HERE')
ax.set_ylabel('YOUR_CODE_HERE')
ax.legend(fontsize=14)
plt.yscale('log')
plt.xscale('log')
# plt.xlim(YOUR_CODE_HERE, YOUR_CODE_HERE)
# plt.ylim(YOUR_CODE_HERE, YOUR_CODE_HERE
ax.grid(True)
plt.show()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5:</b>  

As you may have noticed, the risk in the system does not satisfy your recommendation. Play with the values of $n$ and $p$ to see how you can satisfy the safety standard. Then, select a few "modifications" to the $n$ and/or $p$ values and describe how you might be able to make that change in reality. Keep in mind that making interventions to the system is expensive, so to be realistic you should try to do this by making the <em>smallest</em> change possible; in other words, find the smallest change in $n$ or $p$ separately; don't change many values all at once.

Report your findings as if they were a recommendation to the city. For example: <em>if one were to reduce $p$ of <code>something</code> from <code>value</code> to <code>value</code> by implementing <code>your plan here</code>, the FN curve would shift <code>specify direction</code>, and the safety standard would be satisfied.</em>
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>
Yes, this is an overly simplistic exercise exercise, but it is a good way to think about how to apply the concepts of risk analysis to real-world problems, and to make sure you understand the mathematics of how they are constructed. Also note that each "point" comes from a scenario; in real risk applications it can be quite involved to decide on and carry out the computations required for each scenario.
</p></div>

```python
n_and_p_modified = n_and_p.copy()
n_and_p_modified[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE

DUPLICATE_ANALYSIS_FROM_ABOVE_WITH_MODIFIED_VALUES
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 6:</b>  

Calculate the expected number of people getting sick per year for the factory under the current conditions.

<em>Note that we don't do anything with this calculation in this assignment, but it is often a useful way to quantify the risk of a system, or, for example, comparing two or more types of systemts; various factories, in this case.</em>

</p>
</div>

```python
YOUR_CODE_HERE
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
