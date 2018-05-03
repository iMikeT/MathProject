# Can be implemented in the FE function by replacing the FEM with two other equations

import matplotlib.pyplot as plt
import numpy as np

def HM(f, U0, T, n):
    # Solve u' = f(u,t), u(0) = U0 with n steps
    t = np.zeros(n+1)
    u = np.zeros(n+1)
    u[0] = U0
    t[0] = 0
    dt = T/float(n)
    for k in range(0,n):
        t[k+1] = t[k] + dt
        u_star = u[k] + dt*f(u[k], t[k])
        u[k+1] = u[k] + 0.5*dt*f(u[k], t[k]) + 0.5*dt*f(u_star, t[k+1])
    return u, t

def f(u, t):
    return u # for example u' = u

def test_FEM_Against_Hand_Calc():
    u, t = FEM(f, U0=1, T=0.2, n=2)
    exact = np.array([1, 1.1, 1.21]) # Hand Calculations
    error = np.abs(exact - u).max()
    success = error < 10**(-14)
    assert success, "|exact - u| = %g != 0" % error

u, t = HM(f, U0=1, T=4, n=10)

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
plt.show()