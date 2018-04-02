import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pymc
import seaborn as sns

n, p = 1, 0.4
s = np.random.binomial(n, p, 350) # 100:39, 150:59, 200:84, 250:96, 300:121, 350:142
print(s)
print(sum(s))

theta_real = 0.4
flips = 350
heads = 142

beta_params = [(22, -22), (40, 10)]

dist = stats.beta
x = np.linspace(0, 1, 100)


for (a_prior, b_prior), c in zip(beta_params, ('b', 'g')):
    p_theta_given_y = dist.pdf(x, a_prior + heads, b_prior + flips - heads)
    plt.plot(x, p_theta_given_y, c)
    plt.fill_between(x, 0, p_theta_given_y, color=c, alpha=0.6)

plt.axvline(theta_real, ymax=0.3, color='k')
plt.xlim(-0.1)
plt.ylim(0.12)
plt.xlabel(r'$\theta$')
plt.legend(['Prior', 'Posterior'], loc='upper left')
plt.annotate('{:d} experiments\n{:d} heads'.format(flips,
         heads), xy=(1, 5), xytext=(0.2, 5.4),
            size=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='w'))
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().xaxis.grid(True, linestyle='--')
plt.show()
'''
n, p = 1, 0.4
s = np.random.binomial(n, p, 150)
print(s)
print(sum(s))

trials = [16, 32, 50, 150]
data = [8, 13, 21, 67]

beta_params = [(1, 1), (20, 20)]

dist = stats.beta
x = np.linspace(0, 1, 100)

for idx, N in enumerate(trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
    else:
        plt.subplot(4, 3, idx+3)
    y = data[idx]
    for (a_prior, b_prior), c in zip(beta_params, ('b', 'g')):
        p_theta_given_y = dist.pdf(x, a_prior + y, b_prior + N - y)
        plt.plot(x, p_theta_given_y, c)
        plt.fill_between(x, 0, p_theta_given_y, color=c, alpha=0.6)

    plt.axvline(theta_real, ymax=0.3, color='k')
    plt.plot(0, 0, label="{:d} experiments\n{:d} heads".format(N, y)
             , alpha=0)
    plt.xlim(0,1)
    plt.ylim(0,12)
    plt.xlabel(r'$\theta$')
    plt.legend()
    plt.gca().axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()
'''