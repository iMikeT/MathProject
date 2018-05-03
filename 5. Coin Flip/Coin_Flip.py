import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pymc
import seaborn as sns

 
flips = 50
heads = 18

beta_params = [(7, -7), (25, 25)]

dist = stats.beta
x = np.linspace(0, 1, 100)

for (a_prior, b_prior), c in zip(beta_params, ('b', 'g')):
    p_theta_given_y = dist.pdf(x, a_prior + heads, b_prior + flips - heads)
    plt.plot(x, p_theta_given_y, c)
    plt.fill_between(x, 0, p_theta_given_y, color=c, alpha=0.6)

plt.xlim(0,1)
plt.ylim(0.12)
plt.xlabel(r'$\theta$', fontsize=15)
plt.legend(['Prior', 'Posterior'], fontsize=15, loc='upper left')
plt.annotate('{:d} experiments\n{:d} heads'.format(flips,
         heads), xy=(1, 5), xytext=(0.31, 5.6),
            size=15, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='w'))
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().xaxis.grid(True, linestyle='--')
plt.show()

beta_params1 = [(7, -7), (40, 10)]

dist = stats.beta
x = np.linspace(0, 1, 100)

for (a_prior, b_prior), c in zip(beta_params1, ('b', 'g')):
    p_theta_given_y = dist.pdf(x, a_prior + heads, b_prior + flips - heads)
    plt.plot(x, p_theta_given_y, c)
    plt.fill_between(x, 0, p_theta_given_y, color=c, alpha=0.6)

plt.xlim(0,1)
plt.ylim(0.12)
plt.xlabel(r'$\theta$', fontsize=15)
plt.legend(['Prior', 'Posterior'], fontsize=15, loc='upper left')
plt.annotate('{:d} experiments\n{:d} heads'.format(flips,
         heads), xy=(1, 5), xytext=(0.31, 5.6),
            size=15, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='w'))
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().xaxis.grid(True, linestyle='--')
plt.show()