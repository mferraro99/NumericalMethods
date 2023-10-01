# Integration methods supported:
# 1) Trapezoidal rule (With error calculation)
# 2) Corrected trapezoidal rule 
# 3) Simpson's rule
# 4) Simpson's composite rule

import math as mt
import numpy as np

# Constants definition
e = mt.e
pi = mt.pi
h = 10**-8 # Increment for derivatives

# Function e^(x)*x^2

# Boundaries
a = 0
b = pi
# Number of points -> 128 points? Then N=129 (and so on...)
N = 129

#### Functions

# Calculate the value of the function in a given point
def f(x):
    fx = e**x * mt.cos(x)
    return fx

# Calculate the first derivative in a point using the central 
# difference method
def df(x,h):
    numerator = f(x+h)-f(x-h)
    denominator = 2*h
    return numerator/denominator

# Calculate the second derivative in a point using the central
# difference method 
def d2f(x,h):
    numerator = f(x-h)-2*f(x)+f(x+h)
    denominator = h**2
    return numerator/denominator

# Calculate the third derivative in a point using the central
# difference method 
def d3f(x,h):
    numerator = f(x-2*h)-2*f(x-h)+2*f(x+h)-f(x+2*h)
    denominator = 2*h**3
    return numerator/denominator

#####################################################################
# *** Method 1: Trapezoidal rule (No error correction)***

# The estimated error is determined calculating the second
# derivative in the middle point of every interval using the central
# difference method

# Definition of a linspace
x = np.linspace(a,b,N)

fx = []
for i in x: fx.append(f(i))

d2f_trapez = []
I,Err_asympt = 0,0
for i in range(0,len(fx)-1):

    # In integral
    I_n = (fx[i+1]+fx[i])* 1/2 * (x[i+1]-x[i])
    I += I_n

    # E_n asymptotic error
    Err_n = -1/12 * (x[i+1]-x[i])**2 * (df(x[i+1],h)-df(x[i],h))
    Err_asympt += Err_n

#####################################################################
# *** Method 2: Corrected trapezoidal rule ***

I = 0
for i in range(len(fx)):
    if i == 0 or i == len(fx)-1:
        I += 1/2 * f(x[i])
    else:
        I += f(x[i])

Err = -1/12 * (x[1]-x[0])**2 * (df(b,h)-df(a,h))
I = I * (x[1]-x[0]) + Err

#####################################################################
# *** Method 3: Simpson's rule ***

I = 0
for i in range(len(fx)-1):
    I_n = ((x[i+1]-x[i])/2)/3 * (f(x[i+1])+4*f(1/2*(x[i+1]+x[i]))+f(x[i]))
    I += I_n

print(I)
#####################################################################
# *** Method 4: Simpson's composite rule ***

I = 0
for i in range(len(fx)):

    if i == 0 or i == len(fx)-1:
        c = 1
        print(i,c)
    else:
        if i % 2 != 0: # i is odd
            c = 4
            print(i,c)
        else: # i is even
            c = 2
            print(i,c)
    
    I += c * f(x[i])

I = (x[1]-x[0])/3 * I
print(I)
