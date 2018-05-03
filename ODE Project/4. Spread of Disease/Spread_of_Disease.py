import math
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib as "plt"
"""
Simple case where population is constant with 2 groups; the susceptibles (S)
and the infectives (I) who have the disease and can transmit. the system is given by
S' = -r*S*I
I' = r*S*I - a*I
Here r and a are given constants reflecting the characteristics of the epidemic.
Initial conditions are S(0) = S_0 and I(0) = I_0 where the initial state is assumed
to be known.

For time t = 0 to t = T, time step is dt = T/n
"""
def epidemic(T, n, S_0, I_0, r, a):
    dt = T/float(n)
    t = np.linspace(0, T, n+1)
    S = np.zeros(n+1)
    I = np.zeros(n+1)
    S[0] = S_0
    I[0] = I_0
    for k in range(0, n):
        S[k+1] = S[k] - dt*r*S[k]*I[k]
        I[k+1] = I[k] + dt*(r*S[k]*I[k] - a*I[k])
    return S, I, t

# Initial Data
T = 14
n = 1000
S_0 = 762
I_0 = 1 # One person ill at t = 0
r = 2.18*10**(-3)
a = 0.44

S, I, t = epidemic(T, n, S_0, I_0, r, a)

# Plots

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)
plt.plot(t, S)
plt.xlabel("Time in days")
plt.ylabel("Susceptibles")

fig2 = plt.figure(2)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)
plt.plot(t, I)
plt.xlabel("Time in days")
plt.ylabel("Infectives")

plt.show()
