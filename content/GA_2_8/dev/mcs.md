# Monte Carlo Simulation

Goal is to compute the distribution of profit for a given minute.

Begin with the equation for how much we will profit _given that one of our $N_t$ tickets is the winning ticket_:

$$
B = \frac{W}{N_{w}} - 3N_t
$$

where:
- $B$ is profit (or benefit; we use $B$ to avoid confusion with $P$), in USD
- $W$ is the total payout, in USD
- $N_{w}$ is the number of winning tickets purchased (bt participants) for the winning minute, $m_w$
- $N_t$ is the number of tickets we purchase
- 3 is the cost per ticket (in USD)

Note in particular that the number of (potential) winning tickets purchased bt participants varies (quite significantly!) by minute, which can be denoted as $N_{w}(m)$, where $m$ is the minute.

Because we are choosing which tickets to purchase, we will consider this on a per ticket basis, first as an expected value, then as a distribution (a function of random variables).

## Expected Profit

Following the rules of expectation, we can find the expected profit:

$$
\mathbb{E}\big[B\big] = \mathbb{E}\Bigg[\frac{W}{N_{w}} - 3\Bigg]
$$

Noting that the number of tickets purchased is deterministic, 

$$
\mathbb{E}\big[B\big]
= \frac{\mathbb{E}\big[W\big]}{\mathbb{E}\big[N_{w}\big]} - 3N_t
= \mathbb{E}\big[B(m)\big] - 3N_t
$$

where $\mathbb{E}\big[B(m)\big]$ is the expected winnings from a specific minute $m$ with probability of winning $p(m)$:

$$
\mathbb{E}\big[B(m)\big] = \sum_{m=1}^{N_t} \frac{p(m)\mathbb{E}\big[W\big]}{\mathbb{E}\big[N_{w}(m)\big]}
$$

thus, the expected value requires the computation of probability and expected number of tickets purchased bt participants for each minute.

## Number of Tickets Purchased bt Participants

The number of tickets purchased bt participants, $N_w(m)$, is a function of the minute, $m$. The subscript $w$ is used because in the case that the minute $m$ is the winning minute, $m_w$, the number of tickets purchased bt participants is the number of winning tickets, which has a significant impact on the payout of the winners.

Ideally, this discrete random variable has a distribution that can be modelled from the following:
- day, relative to historic breakup day
- minute, relative to historic breakup minute
- behavioral characteristics of participants (hour, minute)

Behavioral characteristics of participants can only be evaluated with detailed records of actual tickets purchased. The distribution should also take into account the probability of a minute remaining completely unchosen.

Since this is complicated, we will create an empirical distribution, incorporated into our Python code.

## Function of Random Variables

$p_i$ is the probability of a ticket being correct, which will be modelled with two independent Gaussian distributions, one for day, the other for minute. As the payout is variable each year, we will consider this as a random variable, $W$ (but initially ignore this effect). The number of winning tickets, $N_{w}(m)$, is a function of the minute, $m$, and the distribution is described above.

The equations above allow a straightforward calculation of the benefit, however there are two binary elements on which this depends:
1. Did we buy the ticket associated with a specific minute? and,
2. Was the ticket the winning ticket?

We can capture this in a generalized function of random variables using indicator function $I_T(m)$, where $I_T$ is equal to 1 for all minutes $m$ that are included in the minutes defined by a set of tickets, $T$. 

$$
I_T(m) = \begin{cases}
1 & \text{if } m \in T \\
0 & \text{if } m \notin T
\end{cases}
$$

We will use $I_W$ to denote the indicator function for the winning ticket (of which only one, $m_w$ is possible) and $I_T$ to denote the tickets purchased (i.e., $M_t \in T$).

Thus our function of random variables is defined for each ticket purchsed, $m_t$ and winning ticket, $m_w$:

$$
b(m) = \frac{I_W(m) W}{N_{w}(m)} - 3 I_T(m)
$$

The total profit is thus:

$$
B = \int_{-\infty}^{+\infty} b(m) \, dm
$$


$$
B = \sum_{m=1}^{N_t} b(m)
$$

## Monte Carlo Simulation

In theory: easy. In practice: hard to do efficiently.

Key paramters:
- $M_w$ a sample of winning tickets which is found by sampling from the distribution of day and minute and rounding to the nearest minute (integer)
- a realization in $M_w$ is $m_w$

Use the "hat" symbol to denote a sample, e.g., $\hat{M}_w$ is a sample of the winning minutes.

### Brute Force Method

A simple implementation of MCS would proceed as follows:
1. Determine the set of tickets to purchase, $T$
2. Generate $M_w$
3. Compute $B$ for each $m_w$ in $T$ by sampling from $N_{w}(m_w)$
4. Repeat steps 2-3 for all $T$
5. Evaluate the distribution of $B$

However, this is computationally inefficient, as for the majority of samples the winning ticket will not be in the set of tickets purchased, and it is not needed to sample from the distribution of $N_{w}(m)$ for every minute selected.

### Improved Method

Note that the function of random variables can be split in two, where $B=B_1+B_2$, and:

$$
B_1 = \frac{I_T(m_w) W}{N_{w}(m_w)}
$$

which is a function of the winning ticket, and:

$$
B_2 = -3 I_P(m_t) = -3 N_t
$$

which is the cost of the tickets purchased (deterministic).

The algorithm for the Monte Carlo simulation could then be implemented as follows:
1. Determine the set of tickets to purchase, $T$
2. Compute $B_2$
3. Generate $M_w$
4. If $m_w\notin T$, set $B_1=0$
5. If $m_w \in T$, compute $B_1$ by sampling from $N_w(m_w)$
6. Compute $B = B_1 + B_2$
7. Repeat steps 3-5 for all $T$
8. Evaluate the distribution of $B$

In practice, for $ m_w \notin T$ it is not necessary to define $B_1=0$ as this takes up computer memory unnecessarily.


### Consideration of Sample Size

Note that a simple estimate for the number of samples needed is given by

$$
V = \frac{1}{\sqrt{Np}}
$$

where $V$ is the coefficient of variation of $p$. If $p=1e-5$ and $V=0.1$, then the number of samples needed is $N=1e7$.

To sufficiently capture the distribution of $B$ it is necessary to capture the characteristics of $N_{w}(m)$, which is highly variable from minute to minute. This would require a large number of simulations for each purchased minute, $m_t$, considering the fact that popular minutes are purchased up to order 100 times, with the mode reaching order 10. It is also especially important considering the fact that even for the most popular minutes (and thus the most likely to be the winning minute!), the number of winning tickets can be 0!

The sample $\hat{M}_w$ should large enough such that the subset of purchased tickets, $\hat{M}_t\in \hat{M}_w$, is itself sufficiently large enough to characterize the distribution of $B(m_w, m_t)$. This is proportional to the number of tickets purchased, $N_t$. Assuming that order 100 samples are needed to characterize an arbitrary distribution of $N_w(m)$ and that the probability of winning for some of the less likely minutes is order 1e-6 of being the winning minute, then the number of samples needed to ensure a coefficient of variation of 0.1 is $N=N_t \cdot 1e10$. If we by 100 tickets, then we need $N=1e13$ samples, which is very large!

_Can we do importance sampling and just sample every minute, then adjust the samples of the function of random variables by 1 over the ratio of the probability of the minute being the winning minute? This would assume that the "sampling distribution" is simply the $m_w$ being a certain event (probability 1.0)._