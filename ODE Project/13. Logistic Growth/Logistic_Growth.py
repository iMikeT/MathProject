import matplotlib.pyplot as plt
import numpy as np

def f(u, t):
    return alpha*u*(1 - u/R) # Logistic Growth (any function)

alpha = 0.2 # Known constants of the function
R = 4.0
U0 = 0.2
T = 50
n = 200

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

u, t = FEM(f, U0, T, n)

fig1 = plt.figure(1)
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(0) # This rotates each of the x axis variables
ax1.grid(True)

tline = np.linspace(0, T, 1001)
exact = (0.5*(8*np.exp(0.4*tline) - np.sqrt(23104*np.exp(0.4*tline))))/(np.exp(0.4*tline) - 361)

plt.plot(tline, exact, "--", linewidth=2)
plt.plot(t, u, linewidth=2)
plt.xlabel("Value for $t$", fontsize=12)
plt.ylabel("Value for $u$", fontsize=12)
plt.legend(["Exact Solution", "Num. Sol. for $u^{\prime} = %su\\left(1 - \\frac{u}{%s}\\right)$" % (alpha, R)], fontsize=13, borderpad=1)
plt.title("Logistic Growth: $\\alpha$ = %s, $R$ = %s, $\\Delta t$ = %s, n = 200" % (alpha, R, t[1]-t[0]), fontsize=15)
plt.subplots_adjust(left = 0.12, bottom = 0.1, right = 0.92, top = 0.89) # Sets the configure subplot values
# wspace and hspace are for padding for multiple figures
plt.yticks(np.arange(0.2, 4.2, 0.2))
plt.show()