import math
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib as "plt"

# Exponential Growth

def compute_u(u0, T, n):
    # Solve u'(t) = u(t), u(0) = u0 for t in [0, T] with n steps
    u = u0
    dt = T/float(n)
    for k in range(0, n):
        u = (1 + dt)*u
    return u # u(T)

T = 1
n = 5
u0 = 1
print("u(1) =", compute_u(u0, T, n))

# Exponential Growth With Plot

def compute_u(u0, T, n):
    # Solve u'(t) = u(t), u(0) = u0 for t in [0, T] with n steps
    t = np.linspace(0, T, n+1) # From 0 to T with n strips
    t[0] = 0
    u = np.zeros(n+1)
    u[0] = u0
    dt = T/float(n)
    for k in range(0, n):
        u[k+1] = (1 + dt)*u[k]
        t[k+1] = t[k] + dt
    return u, t

T = 1
n = 5
u0 = 1
u, t = compute_u(u0, T, n)
print("u(1) =", u[n])

# Plot
# Figure Customisations

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True) #, color = "g", linestyle = "-", linewidth = 5)# adds a grid. Can change the grid

# Ledgend Titles and Lables
# Come before the show()

tline = np.linspace(0, T, 1001) # For accuracy
v = np.exp(tline) # Very accurate data for exp(x)
plt.plot(tline, v) # Plot of exp(x)
plt.plot(t, u)
plt.xlabel("Value for x")
plt.ylabel("Value for exp(x)")
plt.legend(["Exact Graph of exp(x)", "Approximate Graph of exp(x)"])
plt.title("Approximate and Correct Discrete Functions, n = %d" % n)
plt.subplots_adjust(bottom = 0.13, wspace = 0.2, hspace = 0) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
plt.yticks(np.arange(v.min(), v.max(), 0.1))
plt.show()
