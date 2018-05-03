import matplotlib.pyplot as plt
import numpy as np

def f(u, t):
    return u # for example u' = u

def FEM(f, U0, T, n):
    # Solve u' = f(u,t), u(0) = U0 with n steps
    t = np.zeros(n+1)
    u = np.zeros(n+1)
    u[0] = U0
    t[0] = 0
    dt = T/float(n)
    for k in range(0,n):
        t[k+1] = t[k] + dt
        u[k+1] = u[k] + dt*f(u[k], t[k])
    return u, t

def test_FEM_Against_Hand_Calc():
    u, t = FEM(f, U0=1, T=0.2, n=2)
    exact = np.array([1, 1.1, 1.21]) # Hand Calculations
    error = np.abs(exact - u).max()
    success = error < 10**(-14)
    assert success, "|exact - u| = %g != 0" % error

# If the solution u(t) is linear in t the the FEM will give an exact solution
# u(t) = at + U0 would give f(u,t) = u'(t) = a. We can make f more complex by adding
# something that can be zero like: f(u,t) = a + (u - (at + U0))^4

def test_FEM_Against_Linear_Sol():
    # Use fact that the solution is an exact numerical value for testing
    a = 0.2
    U0 = 3 # Example values given
    def f(u,t):
        return a + (u - u_exact(t))**4

    def u_exact(t):
        return a*t + U0

    u, t = FEM(f, U0 = u_exact(0), T = 3, n = 5)
    u_e = u_exact(t)
    error = np.abs(u_e - u).max()
    success = error < 10**(-14)
    assert success, "|exact - u| = %g != 0" % error

# Continuous Solution

# Say we want to to find u between t[i] and t[i+1]? Like at the midpoint t = t[i] + dt/2
# We can use interpolation to find value u. The simplest is to assume u varies linearly
# on each time interval. On the interval [t[i], t[i+1]] the linear variation of u becomes
# u[t] = u[t[i]] + ((u[t[i+1]] - u[t[i]])*(t - t[i]))/(t[i+1] - t[i])

u, t = FEM(f, U0=1, T=4, n=10)

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)

tline = np.linspace(0, 4, 1001)
exact = np.exp(tline)
plt.plot(tline, exact, "--", linewidth=2)
plt.plot(t, u, linewidth=2)
plt.xlabel("Value for t", fontsize=12)
plt.ylabel("Value for $u^{\prime}(t) = u{t}$", fontsize=12)
plt.legend(["Exact Solution $e^{t}$", "Numerical Solution for $u = u^{\prime}$"], fontsize=12, borderpad=1)
plt.title("Comparison Between Exact and Numerical Solutions, n = 10", fontsize=14)
plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.92, top = 0.89) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
"""
u, t = FEM(f, U0=1, T=4, n=20)

fig2 = plt.figure(2)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)

tline = np.linspace(0, 4, 1001)
exact = np.exp(tline)
plt.plot(tline, exact, "--", linewidth=2)
plt.plot(t, u, linewidth=2)
plt.xlabel("Value for t", fontsize=12)
plt.ylabel("Value for $u^{\prime}(t) = u{t}$", fontsize=12)
plt.legend(["Exact Solution $e^{t}$", "Numerical Solution for $u = u^{\prime}$"], fontsize=12, borderpad=1)
plt.title("Comparison Between Exact and Numerical Solutions, n = 20", fontsize=14)
plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.92, top = 0.89) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
"""

plt.show()

test_FEM_Against_Linear_Sol()