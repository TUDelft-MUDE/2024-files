
import sympy as sym

x, beta, mu = sym.symbols('x beta mu')
F_gumbel = YOUR CODE HERE

Prob_non_exc = sym.symbols('Prob_non_exc')
YOUR CODE HERE
x_sol = YOUR CODE HERE

display(x_sol)

Prob_non_exc_list = [1/773, 0.25, 0.5, 0.75, 772/773]
for i in range(len(Prob_non_exc_list)):
    display(x_sol.subs({beta:13.097, mu:28.167, Prob_non_exc:Prob_non_exc_list[i]}))

f_gumbel = YOUR CODE HERE

display(f_gumbel)

f = YOUR CODE HERE
sym.plot(f,(x,-1,6));

x = sym.symbols('x')
F = YOUR CODE HERE
sym.plot(F, (x,-1,6));

import numpy as np
assert sym.simplify(F_gumbel - sym.exp(-sym.exp((mu - x)/beta))) == 0 , 'Error: Gumbel distribution is not correct'
assert sym.simplify(x_sol + beta*sym.log(sym.exp(-mu/beta)*sym.log(1/Prob_non_exc))) == 0, 'Error: inverted Gumbel distribution is not correct'
assert sym.simplify(f_gumbel - sym.exp(-((x-mu)/beta+sym.exp(-(x-mu)/beta)))/beta) == 0, 'Error: probability densily function is not correct'
assert np.allclose(np.array([0.        , 0.        , 0.05555556, 0.13333333, 0.21111111,       0.28888889, 0.36      , 0.36      , 0.40938272, 1.        ]), sym.lambdify(x, F.rewrite(sym.Piecewise).simplify())(np.linspace(-1,6,10))), 'Error: Piecewise cumulative distribution function is not correct'

