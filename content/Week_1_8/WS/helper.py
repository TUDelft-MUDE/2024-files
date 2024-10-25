import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_contour(bivar, xy_lim, data=None,
                 case=None, region=None,
                 nb_points=200):
    """Contour plot of PDF in the bivariate plane (X,Y).
    
    bivar: bivariate distribution, defined by
        scipy.stats.multivariate_normal.
    xy_lim: list. 1x4 list of xlim and ylim (in order) for plot limits.
        
    Optional arguments:
    
    data: array. A 2xN array of points. If Nx2 data will be transposed.
    case: list. Typically a 1x2 list defining a key
        calculation point; plots as a red dot(s)
    region: array. A 2xN array of points that define the
        boundary of a region that will be shaded in gray between
        the boundary and the max y-axis value (ylim[1]).
    nb_points: int. Size of the grid (default: 200).

    returns: matplotlib.pyplot Figure and Axis objects. 
    """
    
    f, ax = plt.subplots(1)

    xlim = [xy_lim[0], xy_lim[1]]
    ylim = [xy_lim[2], xy_lim[3]]

    x = np.linspace(xlim[0], xlim[1], nb_points)
    y = np.linspace(ylim[0], ylim[1], nb_points)
    X,Y = np.meshgrid(x,y)
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j]>0 and Y[i,j]>0:
                    pdf[i,j] = bivar.pdf([X[i,j], Y[i,j]])
    
    ax.contour(X, Y, pdf, levels=8, cmap=cm.Blues)
    
    if isinstance(data, np.ndarray):
        if data.shape[1] == 2:
             data = data.T
        ax.scatter(data[0,:], data[1,:], s=10.0,
                   facecolors='none', edgecolors='darkgray',
                   label='Data')
    
    if isinstance(region, np.ndarray):
        ax.plot(region[0,:], region[1,:],
                label='Boundary', color='k')
        ax.fill_between(region[0,:], region[1,:],
                        ylim[1], label='Region', color='grey')
    
    if case:
        ax.plot(case[0], case[1], 'ro',
                label='Case')
    
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    if isinstance(region, np.ndarray) or case:
        ax.legend()
    return f, ax 