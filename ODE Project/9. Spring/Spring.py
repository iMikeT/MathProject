# Compute the position of the box as a function of time.
# Then we can compute the velocity, acceleration, spring force
# and damping force.

# Introduce Y(t) as the vertical position of the center point
# of the box. We'll derive an equation that has Y(t) as the solution.
# This equation will be solved by an algorithm, which will be carried
# out by the program.

# Let S denote Stretch where S > 0 is stretch and S < 0 is compressed.
# L is the unstretched length of the spring. So, at time t, the length
# of the spring is L + S(t). Thus, given the position of the plate
# w(t), the length L + S(t) and the height of the box b, the position
# Y(t) is then: Y(t) = w(t) - (L + S(t)) - (b/2)

# We will need to find S(t) to then find the position Y(t).

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos as cos
from numpy import sin as sin

def w(t):
    w = -cos(2*t)
    return w

def Box_Position(S_0, b, L, m, k, B, dt, g, N):
    S = np.zeros(N+1)
    Y = np.zeros(N+1)
    t = 0
    gamma = B*dt/2.0
    S[0] = S_0
    i = 0
    # Special formula for i = 0
    S[i+1] = (1/(2*m))*(2*m*S[i] - dt**2*k*S[i] + m*(w(t+dt) - 2*w(t) + w(t-dt)) +
                        dt**2*m*g)
    t = dt
    for i in range(1,N):
        S[i+1] = (1/(m+gamma))*(2*m*S[i] - m*S[i-1] + gamma*dt*S[i-1] -
                        dt**2*k*S[i] + m*(w(t+dt) - 2*w(t) + w(t-dt)) +
                        dt**2*m*g)
        t += dt
    Y[0] = w(0) - (L + S_0) - (b/2)
    t = dt
    for i in range(0,N):
        Y[i+1] = w(t) - (L + S[i+1]) - (b/2)
        t += dt
    return Y, S

def exact(g, b, m, k, N, S_0):
    y = np.zeros(N+1)
    s = np.zeros(N+1)
    t = np.linspace(0, 8*np.pi, len(S))
    s[0] = S_0
    # -g*(np.cos(t[0]/m) + np.sin(t[0]/m)) + (m*g/k) - 4*np.cos(2*t[0])/3
    y[0] = w(0) - (L + s[0]) - (b/2)
    for i in range(0,N):
    #    S[i+1] = -g*(np.cos(t[i+1]/m) + np.sin(t[i+1]/m)) + (m*g/k) - 4*np.cos(2*t[i+1])/3
        s = (m*g/k) - 4*cos(2*t)/3
        y = w(t) - L - s - b/2
    return  t, y, s

# Set Data

m = 1
b = 2
L = 10
k = 1
B = 0
S_0 = 0
dt = 2*np.pi/40
g = 9.81
N = 80

Y, S = Box_Position(S_0, b, L, m, k, B, dt, g, N)
t, y, s = exact(g,b, m, k, N, S_0)

print(y)
error = abs(s - S)

# Plot
# Figure Customisations

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True) #, color = "g", linestyle = "-", linewidth = 5)# adds a grid. Can change the grid

# Ledgend Titles and Lables
# Come before the show()

plt.plot(t, y)
plt.plot(t, Y)
plt.xlabel("Time t")
plt.ylabel("Position of Box")
plt.legend(["Exact Graph of y(t)", "Approximate Graph of Y(t)"])
plt.title("Approximate and Correct Discrete Functions, N = %d" % N)
plt.subplots_adjust(bottom = 0.13, wspace = 0.2, hspace = 0) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
#plt.yticks(np.arange(v.min(), v.max(), 0.1))

fig2 = plt.figure(2)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True) #, color = "g", linestyle = "-", linewidth = 5)# adds a grid. Can change the grid

# Ledgend Titles and Lables
# Come before the show()

plt.plot(t, error)
plt.xlabel("Time t")
plt.ylabel("Error")
plt.title("Error Between Functions, N = %d" % N)
plt.subplots_adjust(bottom = 0.13, wspace = 0.2, hspace = 0) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
#plt.yticks(np.arange(v.min(), v.max(), 0.1))
plt.show()
