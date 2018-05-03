import matplotlib.pyplot as plt
import numpy as np

def f(u, t):
    return np.array([u[1], -u[0]]) #np.array([u[1], 1./m*(F(t) - beta*u[1] - k*u[0])])  F(t) = 0, beta = 0, k = 1, m = 1

def RungeKutta4(f, U0, T, n):
    t = np.zeros(n+1)
    if isinstance(U0, (float,int)):
        u = np.zeros(n+1)
        K1 = np.zeros(n+1)
        K2 = np.zeros(n+1)
        K3 = np.zeros(n+1)
        K4 = np.zeros(n+1)
    else:
        neq = len(U0)
        u = np.zeros((n+1, neq))
        K1 = np.zeros((n+1, neq))
        K2 = np.zeros((n+1, neq))
        K3 = np.zeros((n+1, neq))
        K4 = np.zeros((n+1, neq))
    u[0] = U0
    t[0] = 0
    dt = T/float(n)
    for k in range(0,n):
        t[k+1] = t[k] + dt
        K1[k] = dt*f(u[k], t[k])
        K2[k] = dt*f(u[k] + 0.5*K1[k], t[k] + 0.5*dt)
        K3[k] = dt*f(u[k] + 0.5*K2[k], t[k] + 0.5*dt)
        K4[k] = dt*f(u[k] + K3[k], t[k] + dt)
        u[k+1] = u[k] + (1/6)*(K1[k] + 2*K2[k] + 2*K3[k] + K4[k])
    return u, t

U0 = np.array([0, 1]) # Known constants of the function
T = 8*np.pi
n = 160

u, t = RungeKutta4(f, U0, T, n)
u0 = u[:,0] # takes all the values from u[0] and stores them as a 1xn array

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)

tline = np.linspace(0, T, 1001)
exact = np.sin(tline)

plt.plot(t, u0, "r-", linewidth=2)
plt.plot(tline, np.sin(tline), "b--", linewidth=2)
plt.xlabel("Value for $t$", fontsize=12)
plt.ylabel("Value for $u$", fontsize=12)
plt.legend(["4th-Order Runge-Kutta, n = %d" % n, "Exact"], loc="upper left", fontsize=11, borderpad=1)
plt.title("Approximation of solution u(t) = sin(t) against exact solution")
plt.subplots_adjust(left = 0.12, bottom = 0.1, right = 0.92, top = 0.89) # Sets the configure subplot values
plt.yticks(np.arange(-1.5, 2.5, 0.5))

plt.show()