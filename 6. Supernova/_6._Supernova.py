import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import corner
import numpy as np
from math import log10, sqrt
from scipy.integrate import quad
from datetime import datetime
import pymc

path = 'F:\\Documents\\University\\Math Project\\Data Analysis\\6. Supernova\\Data.txt'

coordinates = pd.read_csv(path, 
                      delim_whitespace=True, 
                     names=['x', 'y'])

xdata = coordinates['x'].as_matrix()
ydata = coordinates['y'].as_matrix()

my_cov1 = np.array([[21282, -10840, 1918, 451, 946, 614, 785, 686, 581, 233, 881, 133, 475, 295, 277, 282, 412, 293, 337, 278, 219, 297, 156, 235, 133, 179, -25, -106, 0, 137, 168], #1
                    [0, 28155, -2217, 1702, 74, 322, 380, 273, 424, 487, 266, 303, 406, 468, 447, 398, 464, 403, 455, 468, 417, 444, 351, 399, 83, 167, -86, 15, -2, 76, 243], #2
                    [0, 0, 6162, -1593, 1463, 419, 715, 580, 664, 465, 613, 268, 570, 376, 405, 352, 456, 340, 412, 355, 317, 341, 242, 289, 119, 152, -69, -33, -44, 37, 209], #3
                    [0, 0, 0, 5235, -722, 776, 588, 591, 583, 403, 651, 212, 555, 353, 355, 323, 442, 319, 372, 337, 288, 343, 210, 272, 92, 167, -48, -29, -21, 50, 229], #4
                    [0, 0, 0, 0, 7303, -508, 1026, 514, 596, 315, 621, 247, 493, 320, 375, 290, 383, 286, 350, 300, 269, 313, 198, 251, 99, 126, 18, 46, 13, 10, 203], #5
                    [0, 0, 0, 0, 0, 3150, -249, 800, 431, 358, 414, 173, 514, 231, 248, 221, 293, 187, 245, 198, 175, 231, 126, 210, 103, 170, 51, 66, -8, -51, 308], #6
                    [0, 0, 0, 0, 0, 0, 3729, -88, 730, 321, 592, 188, 546, 316, 342, 290, 389, 267, 341, 285, 252, 301, 189, 242, 122, 159, 35, 72, 30, 28, 255], #7
                    [0, 0, 0, 0, 0, 0, 0, 3222, -143, 568, 421, 203, 491, 257, 280, 240, 301, 221, 275, 227, 210, 249, 148, 220, 123, 160, 43, 69, 27, 7, 253], #8
                    [0, 0, 0, 0, 0, 0, 0, 0, 3225, -508, 774, 156, 502, 273, 323, 276, 370, 260, 316, 273, 231, 273, 171, 226, 111, 154, 0, 29, 19, 23, 206], #9
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5646, -1735, 691, 295, 362, 316, 305, 370, 280, 346, 313, 276, 310, 217, 274, 131, 175, 38, 118, 78, 48, 303], #10
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8630, -1642, 944, 152, 253, 184, 274, 202, 254, 233, 196, 237, 156, 207, 27, 115, -32, 7, -15, 0, 176], #11
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3855, -754, 502, 225, 278, 294, 274, 285, 253, 239, 255, 173, 229, 181, 177, 93, 124, 132, 108, 227], #12
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4340, -634, 660, 240, 411, 256, 326, 276, 235, 290, 184, 256, 135, 222, 90, 152, 67, 17, 318], #13
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2986, -514, 479, 340, 363, 377, 362, 315, 343, 265, 311, 144, 198, 17, 62, 86, 147, 226], #14
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3592, -134, 606, 333, 422, 374, 333, 349, 267, 300, 157, 184, 9, 71, 85, 136, 202], #15
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1401, 22, 431, 343, 349, 302, 322, 245, 284, 171, 186, 70, 70, 93, 142, 202], #16
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1491, 141, 506, 386, 356, 394, 278, 306, 188, 212, 79, 71, 106, 145, 240], #17
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1203, 200, 435, 331, 379, 281, 311, 184, 209, 49, 51, 110, 197, 181], #18
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1032, 258, 408, 398, 305, 330, 197, 223, 78, 79, 113, 174, 225], #19
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1086, 232, 453, 298, 328, 120, 189, -48, 22, 42, 142, 204], #20
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1006, 151, 329, 282, 169, 195, 58, 80, 95, 192, 188], #21
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1541, 124, 400, 199, 261, 150, 166, 202, 251, 251], #22
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1127, 72, 227, 222, 93, 118, 93, 171, 161], #23
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1723, -105, 406, -3, 180, 190, 198, 247], #24
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1550, 144, 946, 502, 647, 437, 215], #25
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1292, 187, 524, 393, 387, 284], #26
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3941, 587, 1657, 641, 346], #27
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2980, 360, 1124, 305], #28
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4465, -1891, 713], #29
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23902, -1826], #30
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19169]]) #31
# Array of the digonal
sig_squared = [21282, 28155, 6162, 5235, 7303, 3150, 3729, 3222, 3225, 5646, 8630, 3855, 4340, 2986, 3592, 1401, 1491, 1203, 1032, 1086, 1006, 1541, 1127, 1723, 1550, 1292, 3941, 2980, 4465, 23902, 19169]
sig1 = list(map(lambda x: x*10**(-6), sig_squared))
sig2 = list(map(sqrt, sig1))
sig = pd.DataFrame({'sigma':sig2})
edata = sig['sigma'].as_matrix()

my_covT = np.transpose(my_cov1)
for i in range(31):
    for k in range(31):
        if i == k:
            my_covT[i,k] = 0
my_cov = my_cov1 + my_covT
for i in range(31):
    for k in range(31):
        if i == k:
            my_cov[i,k] = sig_squared[i] 
# Now we have full symmetric matrix
my_cov = my_cov*(10**(-6)) # Multiply by 10^-6

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           yerr=sig['sigma'],
           color='k',
           ecolor='Blue',
           linewidth=0.5,
           elinewidth=1,
           capthick=1,
           capsize=3,
           marker='.', 
           ms=5)

#plt.xscale('log')
plt.legend(['Data Points'], fontsize=12, loc='upper left')
plt.ylabel(r'Distance Modulus $\mu_b$', fontsize=12)
plt.xlabel(r'Redshift $z$', fontsize=12)
plt.title('Plot of Observations', fontsize=15)
plt.show()

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.grid(True)
plt.errorbar(coordinates['x'], coordinates['y'], 
           yerr=sig['sigma'],
           color='k',
           ecolor='Blue',
           linewidth=0.5,
           elinewidth=1,
           capthick=1,
           capsize=3,
           marker='.', 
           ms=5)

plt.xscale('log')
plt.legend(['Data Points'], fontsize=12, loc='upper left')
plt.ylabel(r'Distance Modulus $\mu_b$', fontsize=12)
plt.xlabel(r'Redshift $z$', fontsize=12)
plt.title('Plot of Observations', fontsize=15)
plt.show()

# Create some convenience routines for plotting
 
def compute_sigma_level(trace1, trace2, nbins=20):
    # From a set of traces, bin by number of standard deviations
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
 

def plot_MCMC_trace0(xdata, ydata, trace, scatter=False, **kwargs):
    # Plot traces and contours
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[2]) 
    fig2 = plt.figure(2) 
    ax2 = plt.subplot2grid((1,1), (0,0))
    ax2.grid(True) 
    plt.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter: 
        plt.plot(trace[0], trace[2], ',b', alpha=0.1)
    plt.xlabel(r'$h$', fontsize=20) 
    plt.ylabel(r'$\Omega_M$', rotation=0, fontsize=20)
    #plt.title('Error Ellipse', fontsize=15) 
    plt.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.92, top = 0.89) # Sets the configure subplot values
    plt.show()

def plot_MCMC_trace1(xdata, ydata, trace, scatter=False, **kwargs):
    # Plot traces and contours
    xbins, ybins, sigma = compute_sigma_level(trace[1], trace[2]) 
    fig2 = plt.figure(2) 
    ax2 = plt.subplot2grid((1,1), (0,0))
    ax2.grid(True) 
    plt.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter: 
        plt.plot(trace[1], trace[2], ',b', alpha=0.1)
    plt.xlabel(r'$\Omega_\Lambda$', fontsize=20)
    ax2.tick_params(labelleft='off')
    #plt.ylabel(r'Value for $\Omega_M$', fontsize=12)
    #plt.title('Error Ellipse', fontsize=15) 
    plt.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.92, top = 0.89) # Sets the configure subplot values
    plt.show()

def plot_MCMC_trace2(xdata, ydata, trace, scatter=False, **kwargs):
    # Plot traces and contours
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1]) 
    fig2 = plt.figure(2) 
    ax2 = plt.subplot2grid((1,1), (0,0))
    ax2.grid(True) 
    plt.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter: 
        plt.plot(trace[0], trace[1], ',b', alpha=0.1)
    #plt.xlabel(r'Value for $h$', fontsize=12) 
    plt.ylabel(r'$\Omega_\Lambda$', rotation=0, fontsize=20)
    ax2.tick_params(labelbottom='off')
    #plt.title('Error Ellipse', fontsize=15) 
    plt.subplots_adjust(left = 0.16, bottom = 0.11, right = 0.92, top = 0.89) # Sets the configure subplot values
    plt.show()

def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    # Plot both the trace and the model together
    plot_MCMC_trace0(xdata, ydata, trace, True, colors=colors)
    plot_MCMC_trace1(xdata, ydata, trace, True, colors=colors)
    plot_MCMC_trace2(xdata, ydata, trace, True, colors=colors)

# Define the variables needed for the routine, with their prior distributions

# Data from Planck Mission March 21, 2013
Omega_Lambda = pymc.Normal('Omega_Lambda', 0.65, 100) #
Omega_M = pymc.Normal('Omega_M', 0.25, 100) #
h = pymc.Normal('h', 0.65, 100)

# Define the form of the model and likelihood
@pymc.deterministic
def y_model(xdata=xdata, Omega_Lambda=Omega_Lambda, Omega_M=Omega_M, h=h):
    mu_b = [0] * len(xdata)
    if Omega_M<0.05:
        Omega_M=0.05
    if Omega_M>0.45:
        Omega_M=0.45
    if Omega_Lambda<0.45:
        Omega_Lambda=0.45
    if Omega_Lambda>0.85:
        Omega_Lambda=0.85
    def Formula(x):
        root= sqrt(Omega_Lambda + Omega_M*(1 + x)**3 + (1 - Omega_Lambda - Omega_M)*(1 + x)**2)
        result = 1/root                
        return result
    for i in range(len(xdata)):
        z = xdata[i]
        D_L = ((2998/h)*(1 + z))*quad(Formula, 0, z)[0]
        mu_b[i] = 25 + 5*log10(abs(D_L))
    return mu_b
# (We define the error as 2 std)
T = np.linalg.inv(my_cov) # tau = precision matrix = inverse of covariance matrixy = pymc.MvNormal('y', mu=y_model, tau=T, observed=True, value=ydata)
# package the full model in a dictionary
model1 = dict(h=h, Omega_Lambda=Omega_Lambda, Omega_M=Omega_M, y=y)
# run the basic MCMC:
S = pymc.MCMC(model1)
S.sample(iter=100000, burn=80000)
# extract the traces and plot the results
pymc_trace_unifo = [S.trace('h')[:],
                    S.trace('Omega_Lambda')[:],
                    S.trace('Omega_M')[:]]

plot_MCMC_results(xdata, ydata, pymc_trace_unifo)
print()
print("h mean = {:.4f}".format(pymc_trace_unifo[0].mean()))
print("Omega_Lambda mean = {:.4f}".format(pymc_trace_unifo[1].mean()))
print("Omega_M mean = {:.4f}".format(pymc_trace_unifo[2].mean()))
print("h std = {:.4f}".format(pymc_trace_unifo[0].std()))
print("Omega_Lambda std = {:.4f}".format(pymc_trace_unifo[1].std()))
print("Omega_M std = {:.4f}".format(pymc_trace_unifo[2].std()))

fig3 = plt.figure(3)
ax3 = plt.subplot2grid((1,1), (0,0))
ax3.grid(True)

mu = pymc_trace_unifo[0].mean() # mean of distribution
sigma = pymc_trace_unifo[0].std() # standard deviation of distribution
x = pymc_trace_unifo[0]

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='lightgray', edgecolor='black', linewidth=0.5)
# add a 'best fit' line
g = mlab.normpdf(bins, mu, sigma) 
p = mlab.normpdf(bins, 0.65, 0.1)
plt.axvline(x=mu, ymin=0, ymax = 10, linewidth=2, color='blue', linestyle='--')
plt.plot(bins, g, 'r--')
plt.plot(bins, p, 'g--')
plt.gca().axes.get_yaxis().set_visible(False)
plt.ylabel(r'$h$', rotation=0, fontsize=20)
plt.legend(['Mean','Posterior', 'Prior'], fontsize=12, loc='upper left')
ax3.tick_params(labelbottom='off')
plt.show()

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
p = mlab.normpdf(bins, 0.65, 0.1)
plt.axvline(x=mu, ymin=0, ymax = 10, linewidth=2, color='blue', linestyle='--')
plt.plot(bins, g, 'r--')
plt.plot(bins, p, 'g--')
plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(['Mean','Posterior', 'Prior'], fontsize=12, loc='upper left')
ax4.tick_params(labelbottom='off')

plt.show()

fig5 = plt.figure(5)
ax5 = plt.subplot2grid((1,1), (0,0))
ax5.grid(True)

mu = pymc_trace_unifo[2].mean() # mean of distribution
sigma = pymc_trace_unifo[2].std() # standard deviation of distribution
x = pymc_trace_unifo[2]

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='lightgray', edgecolor='black', linewidth=0.5)
# add a 'best fit' line
g = mlab.normpdf(bins, mu, sigma) 
p = mlab.normpdf(bins, 0.25, 0.1)
plt.axvline(x=mu, ymin=0, ymax = 10, linewidth=2, color='blue', linestyle='--')
plt.plot(bins, g, 'r--')
plt.plot(bins, p, 'g--')
plt.xlabel(r'$\Omega_M$', fontsize=20)
plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(['Mean','Posterior', 'Prior'], fontsize=12, loc='upper left')

plt.show()