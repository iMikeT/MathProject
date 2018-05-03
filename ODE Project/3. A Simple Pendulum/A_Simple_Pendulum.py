import math
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib as "plt"

# A Simple Pendulum

# The differential equation is theta''(t) + alpha*sin(theta) = 0
# theta is angle rod makes with vertical, alpha = g/L
# To transform into a system of two first-order equations, we introduce
# a new variable v(t) = theta'(t) (angular velocity) which yields: v'(t) + alpha*sin(theta) = 0
# We have the relation v = theta'(t). This means that the differential
# equation is equivalent to the following system:
#   theta'(t) = v(t)
#       v'(t) = -alpha*sin(theta)
# where initial conditions: theta(0) = theta_0, v(0) = v_0
# Assume that these are given. Common to group the unknowns into vectors
# (theta(t), v(t)) and (theta_0, v_0)

def pendulum(T, n, theta_0, v_0, alpha):
    # Return the motion (theta, v, t) of a pendulum
    dt = T/float(n)
    t = np.linspace(0, T, n+1) # From 0 to T > 0 with time-steps n
    v = np.zeros(n+1) # Create blank list with n+1 elements
    theta = np.zeros(n+1)
    v[0] = v_0             # Overwrite each element
    theta[0] = theta_0
    for k in range(0, n):
        theta[k+1] = theta[k] + dt*v[k]
        v[k+1] = v[k] - alpha*dt*np.sin(theta[k+1])
    return theta, v, t

# Sample Data

theta_0 = np.pi/6
v_0 = 0
alpha = 5
T = 10
n = 1000
theta, v, t = pendulum(T, n, theta_0, v_0, alpha)

# Plots

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)
plt.plot(t, v)
plt.xlabel("Time t")
plt.ylabel("Angular Velocity ${\\nu}(t)$ = ${\Theta}^{\prime}(t)$")

fig2 = plt.figure(2)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)
plt.plot(t, theta, color="g")
plt.xlabel("Time t")
plt.ylabel("Angle ${\Theta}(t)$")

plt.show()
