import numpy as np

# The Simplest Case
def f(t):
    return t*np.exp(t*t)

def integrate(f, T, n, u0):
    h = T/float(n)
    t = np.linspace(0, T, n+1) # From 0 to T with n strips
    I = f(t[0])
    for k in range(1, n): # Goes from 1 to n-1
        I += 2*f(t[k]) # Sum of each element in f from 1 to n-1
    I += f(t[-1]) # Previous sum plus last element n
    I *= (h/2) # Multiply sum 
    I += u0 # Previous sum plus the initial condition
    return float(I)

n = 1000
T = 2
u0 = 0

print("Numerical Solution of t*exp(t*t) is:", integrate(f, T, n, u0))
