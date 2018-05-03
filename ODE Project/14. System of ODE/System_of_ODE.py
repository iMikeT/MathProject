import matplotlib.pyplot as plt
import numpy as np

def f(u, t):
    return np.array([u[1], -u[0]]) #np.array([u[1], 1./m*(F(t) - beta*u[1] - k*u[0])])  F(t) = 0, beta = 0, k = 1, m = 1

def FEM(f, U0, T, n):
    # Solve u' = f(u,t), u(0) = U0 with n steps
    t = np.zeros(n+1)
    if isinstance(U0, (float,int)):
        u = np.zeros(n+1)
    else:
        neq = len(U0)
        u = np.zeros((n+1, neq))
    u[0] = U0
    t[0] = 0
    dt = T/float(n)
    for k in range(0,n):
        t[k+1] = t[k] + dt
        u[k+1] = u[k] + dt*f(u[k], t[k])
    return u, t

U0 = np.array([0, 1]) # Known constants of the function
T = 8*np.pi
n = 400

u, t = FEM(f, U0, T, n)
u0 = u[:,0] # takes all the values from u[0] and stores them as a 1xn array

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)

plt.plot(t, u0, "r-", t, np.sin(t), "b--", linewidth=2)
plt.xlabel("Value for $t$", fontsize=12)
plt.ylabel("Value for $u$", fontsize=12)
plt.legend(["Forward Euler, n = %d" % n, "Exact"], loc="upper left", fontsize=11, borderpad=1)
plt.title("Approximation of solution u(t) = sin(t) against exact solution")
plt.subplots_adjust(left = 0.12, bottom = 0.1, right = 0.92, top = 0.89) # Sets the configure subplot values

plt.show()