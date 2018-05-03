import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from datetime import datetime
import pymc

path = 'F:\\Documents\\University\\Math Project\\Data Analysis\\3. Mystery Linear\\mystery_linear.txt'

coordinates = pd.read_csv(path, 
                      delim_whitespace=True, 
                     names=['x', 'y'])

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           color='Blue',
           ls='None',  
           elinewidth=1,
           capthick=1.5,
           marker='.', 
           ms=6)

plt.legend(['Data'], fontsize=12, loc='upper right')
plt.ylabel(r'Value for $y$', fontsize=12)
plt.xlabel(r'Value for $x$', fontsize=12)
plt.title('Plot of Data Points', fontsize=15)
plt.show()

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           color='Blue',
           ls='None',  
           elinewidth=1,
           capthick=1.5,
           marker='.', 
           ms=6)

x = np.linspace(-8, 8, 100)
y = 17.4 - x*0.5
plt.axvline(-3, color='c')
plt.axhline(19, color='g')
plt.axvline(3, color='m')
plt.axhline(16, color='y')
plt.plot(x, y, color='r')
plt.xlim(-4, 8)
plt.ylim(14.5, 20.5)
plt.xlabel(r'Value for $x$', fontsize=12)
plt.ylabel(r'Value for $y$', fontsize=12)
plt.legend([r'$x = -3$',r'$y = 19$',r'$x = 3$',r'$y = 16$',r'$y = 17.4 - 0.5x$','Data'], fontsize=12, loc='right')
plt.title('Approximation of Linear Solution', fontsize=15)
plt.show()

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           color='Blue',
           ls='None',  
           elinewidth=1,
           capthick=1.5,
           marker='.', 
           ms=6)

x = np.linspace(-3, 3, 100)
y = 17.4 - x*0
plt.plot(x, y, color='r')
plt.xlabel(r'Value for $x$', fontsize=12)
plt.ylabel(r'Value for $y$', fontsize=12)
plt.legend([r'$y = 17.4$', 'Data'], fontsize=12, loc='upper right')
plt.title('Approximation of Gradient', fontsize=15)
plt.show()

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           color='Blue',
           ls='None',  
           elinewidth=1,
           capthick=1.5,
           marker='.', 
           ms=6)

x = np.linspace(-3, 3, 100)
y = 17.4 - x
plt.plot(x, y, color='r')
plt.xlabel(r'Value for $x$', fontsize=12)
plt.ylabel(r'Value for $y$', fontsize=12)
plt.legend([r'$y = 17.4 - x$', 'Data'], fontsize=12, loc='upper right')
plt.title('Approximation of Gradient', fontsize=15)
plt.show()

xdata = coordinates['x'].as_matrix()
ydata = coordinates['y'].as_matrix()
edata = 1.* np.ones_like(xdata)

# Create some convenience routines for plotting
 
def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)
 
    shape = L.shape
    L = L.ravel()
 
    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)
 
    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
     
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])
    
    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)
 
 
def plot_MCMC_trace(xdata, ydata, trace, scatter=False, **kwargs):
    # Plot traces and contours
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    fig2 = plt.figure(2)
    ax2 = plt.subplot2grid((1,1), (0,0))
    ax2.grid(True)
    plt.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        plt.plot(trace[0], trace[1], ',b', alpha=0.1)
    plt.xlabel(r'Value for $c$', fontsize=12)
    plt.ylabel(r'Value for $m$', fontsize=12)
    plt.title('Error Ellipse of the Intercept and the Slope', fontsize=15)
    plt.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.92, top = 0.89) # Sets the configure subplot values

     
def plot_MCMC_model(xdata, ydata, trace):
    # Plot the linear model and 2sigma contours
    fig3 = plt.figure(3)
    ax3 = plt.subplot2grid((1,1), (0,0))
    ax3.grid(True)
    plt.plot(xdata, ydata, '.', Color='Blue', label='Data')

    c, m = trace[:2]
    xfit = np.linspace(-3, 3, 100)
    yfit = c[:,None] + m[:,None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)
 
    plt.plot(xfit, mu, 'r-', linewidth=2, label='pymc')
    plt.fill_between(xfit, mu - sig, mu + sig,
                     color='lightgray', alpha=0.5,
                     label='Uncertainty')
 
    plt.xlabel(r'Value for $x$', fontsize=12)
    plt.ylabel(r'Value for $y$', fontsize=12)
    plt.legend(loc='upper right', numpoints=1,fontsize=12)
    plt.title('Approximation of Linear Solution', fontsize=15)
 

def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    # Plot both the trace and the model together
    plot_MCMC_trace(xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(xdata, ydata, trace)


# Define the variables needed for the routine, with their prior distributions
m = pymc.Uniform('m', -1, 0)
c = pymc.Uniform('c', 16, 18)
 
# Define the form of the model and likelihood
@pymc.deterministic
def y_model(xdata=xdata, c=c, m=m):
    return c + m * xdata
# (We define the error as 2 std)
y = pymc.Normal('y', mu=y_model, tau=1. / (edata/2) ** 2, observed=True, value=ydata)

# package the full model in a dictionary
model1 = dict(c=c, m=m, y_model=y_model, y=y)

# run the basic MCMC:
S = pymc.MCMC(model1)
S.sample(iter=100000, burn=50000)
 
# extract the traces and plot the results
pymc_trace_unifo = [S.trace('c')[:],
              S.trace('m')[:]]
 
plot_MCMC_results(xdata, ydata, pymc_trace_unifo)
print()
print("c mean= {:.4f}".format(pymc_trace_unifo[0].mean()))
print("m mean= {:.4f}".format(pymc_trace_unifo[1].mean()))
print("c std= {:.4f}".format(pymc_trace_unifo[0].std()))
print("m std= {:.4f}".format(pymc_trace_unifo[1].std()))


# Gradient Histogram Plot
fig4 = plt.figure(4)
ax4 = plt.subplot2grid((1,1), (0,0))
ax4.grid(True)

mu = pymc_trace_unifo[1].mean() # mean of distribution
sigma = pymc_trace_unifo[1].std() # standard deviation of distribution
x = pymc_trace_unifo[1]

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='lightgray', edgecolor='black', linewidth=0.5)
# add a 'best fit' line
g = mlab.normpdf(bins, mu, sigma)
plt.axvline(x=mu, ymin=0, ymax = 10, linewidth=2, color='blue', linestyle='--')
plt.plot(bins, g, 'r--')
plt.xlabel(r'$m$', fontsize=12)
plt.gca().axes.get_yaxis().set_visible(False)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.show()

#Intercept Histogram Plot
fig4 = plt.figure(4)
ax4 = plt.subplot2grid((1,1), (0,0))
ax4.grid(True)

mu = pymc_trace_unifo[0].mean() # mean of distribution
sigma = pymc_trace_unifo[0].std() # standard deviation of distribution
x = pymc_trace_unifo[0]

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='lightgray', edgecolor='black', linewidth=0.5)
# add a 'best fit' line
g = mlab.normpdf(bins, mu, sigma)
plt.axvline(x=mu, ymin=0, ymax = 10, linewidth=2, color='blue', linestyle='--')
plt.plot(bins, g, 'r--')
plt.xlabel(r'$c$', fontsize=12)
plt.gca().axes.get_yaxis().set_visible(False)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.show()