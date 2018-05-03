import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from datetime import datetime
import pymc

path = 'F:\\Documents\\University\\Math Project\\Data Analysis\\2. Climate_Change\\co2_gr_gl.txt'

co2_gr = pd.read_csv(path, 
                      delim_whitespace=True, 
                     skiprows=62, 
                     names=['year', 'rate', 'err'])

x = co2_gr['year'].as_matrix()  
y = co2_gr['rate'].as_matrix()
y_error = co2_gr['err'].as_matrix()


plt.figure(figsize=(8,4))
ax = plt.subplot2grid((1,1), (0,0))
ax.grid(True)
plt.title('Growth rate since 1960', fontsize=15)
plt.errorbar(x,y,yerr=y_error,             
        color='Blue', 
        ecolor='k', 
        ls='None',  
        elinewidth=1, 
        capthick=1,
        capsize=3,
        marker='.', 
        ms=5, 
        label='Observed')
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO$_2$ growth rate (ppm/yr)', fontsize=12)
plt.legend(loc=2, numpoints=1, fontsize=12)
plt.show()

def model(x, y): 
    slope = pymc.Normal('slope', 0.1, 1.0)
    intercept = pymc.Normal('intercept', -50.0, 10.0)
    @pymc.deterministic(plot=False)
    def linear(x=x, slope=slope, intercept=intercept):
        return x * slope + intercept
    f = pymc.Normal('f', mu=linear, tau=1.0/y_error, value=y, observed=True)
    return locals()
from pymc import Matplot as mcplt
MDL = pymc.MCMC(model(x,y))
MDL.sample(5e5, 5e4, 100)
pymc_trace_unifo = [MDL.trace('slope')[:],
                    MDL.trace('intercept')[:]]

y_min = MDL.stats()['linear']['quantiles'][2.5]
y_max = MDL.stats()['linear']['quantiles'][97.5]
y_fit = MDL.stats()['linear']['mean']
slope = MDL.stats()['slope']['mean']
slope_err = MDL.stats()['slope']['standard deviation']
intercept = MDL.stats()['intercept']['mean']
intercept_err = MDL.stats()['intercept']['standard deviation']

print()
print('slope:{0:.3f}, intercept:{1:.2f}'.format(slope, intercept))

MDL.stats(['intercept','slope'])

plt.figure(figsize=(8,4))
ax = plt.subplot2grid((1,1), (0,0))
ax.grid(True)
plt.title('Growth rate since 1960', fontsize=15)
plt.errorbar(x,y,yerr=y_error,             
        color='Blue', 
        ecolor='k', 
        ls='None',  
        elinewidth=1, 
        capthick=1,
        capsize=3,
        marker='.', 
        ms=5, 
        label='Observed')
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO$_2$ growth rate (ppm/yr)', fontsize=12)
plt.plot(x, y_fit, 
        'r-', lw=2, label='pymc')
plt.fill_between(x, y_min, y_max, 
        color='lightgray', alpha=0.5, 
        label='Uncertainty')
plt.legend(loc=2, numpoints=1, fontsize=12)

fig5 = plt.figure(5)
ax5 = plt.subplot2grid((1,1), (0,0))
ax5.grid(True)

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
plt.xlabel('slope', fontsize=12)
plt.gca().axes.get_yaxis().set_visible(False)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

fig6 = plt.figure(6)
ax5 = plt.subplot2grid((1,1), (0,0))
ax5.grid(True)

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
plt.xlabel('intercept', fontsize=12)
plt.gca().axes.get_yaxis().set_visible(False)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()