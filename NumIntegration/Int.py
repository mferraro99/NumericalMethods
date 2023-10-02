# Integration methods supported:
# 1) Trapezoidal rule (With error calculation)
# 2) Corrected trapezoidal rule 
# 3) Simpson's rule
# 4) Simpson's composite rule

import math as mt
import numpy as np
import sys

# Define the type of function (continous or pointwise)
# C -> Continous , P -> Pointwise
function_type = "P"

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

if function_type == "C":
    # Definition of a linspace
    x = np.linspace(a,b,N)

    fx = []
    for i in x: fx.append(f(i))
elif function_type == "P":
    x = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]
    fx = [1,0.848746340935915,0.420539362477453,-0.152029664121105,-0.687104063406503,-1.01125325225677,-1.01561050899981,-0.692309483600528,-0.139063079775502,0.470236502645016,0.940095362324134,1.11606908767892,0.935432560402321,0.449274746855092,-0.191688956690769,-0.783741299959972,-1.13483379067851,-1.12679813036417,-0.754972626282178,-0.131999545590381,0.546155822366518,1.06176162492308,1.24514368635754,1.03041166293532,0.478818999800278,-0.238435611203656,-0.893234490342358,-1.2730066804916,-1.24964050010305,-0.822532482595037,-0.121357957050111,0.633144496979515,1.19856770059558,1.38862503623987,1.13440046262079,0.508959360844196,-0.293362081689522,-1.01722100875906,-1.42743682936209,-1.38529333543712,-0.895252333031574,-0.106406166606706,0.732713940699982,1.35233226600296,1.54805838084076,1.24816658264991,0.539422405231801,-0.357715634877373,-1.15753913305342,-1.59997223728802,-1.53501818932001,-0.973388322424015,-0.0862919402245434,0.846572133867687,1.52508515496761,1.72514686438775,1.37253025869948,0.569862926998684,-0.432919060902164,-1.31625185947987,-1.79266329291176,-1.70019090992389,-1.05718413373545,-0.0600252858012934,0.976648186516986,1.71909102638594,1.92176664814234,1.50836585274007,0.599851086958079,-0.520594033891802,-1.49567344548861,-2.0077839972006,-1.88231058869747,-1.14686452489547,-0.026458353172784,1.12511986424234,1.93687558314178,2.13998333488864,1.65660298491479,0.628857577771565,-0.622587448895515,-1.69839898185391,-2.24785529294854,-2.08300899249957,-1.2426275389889,0.0157374117097773,1.29444442163959,2.18125460430814,2.38206979456544,1.81822717372806,0.656236524930966,-0.741001100463102,-1.92733732740418,-2.51567069311324,-2.30406046745783,-1.3446351699197,0.0680972156769864,1.48739312746671,2.45536607825544,2.65052548736682,1.9942798546275]
else:
    print("Error: no type of function defined (Continous or Pointwise)!")
    sys.exit()

#####################################################################
# *** Method 1: Trapezoidal rule (No error correction)***

# The estimated error is determined calculating the second
# derivative in the middle point of every interval using the central
# difference method

I,Err_asympt = 0,0
for i in range(0,len(fx)-1):

    # In integral
    I_n = (fx[i+1]+fx[i])* 1/2 * (x[i+1]-x[i])
    I += I_n

    # E_n asymptotic error
    Err_n = -1/12 * (x[i+1]-x[i])**2 * (df(x[i+1],h)-df(x[i],h))
    Err_asympt += Err_n

print("Number of points: "+str(N-1)+"\n")
print("1) Trapezoidal: "+str(I)+"\n")
#####################################################################
# *** Method 2: Corrected trapezoidal rule ***

I = 0
for i in range(len(fx)):
    if i == 0 or i == len(fx)-1:
        I += 1/2 * f(x[i])
    else:
        I += f(x[i])

Err_comptrap = -1/12 * (x[1]-x[0])**2 * (df(b,h)-df(a,h))
I = I * (x[1]-x[0]) + Err_comptrap

print("2) Corrected trapezoidal: "+str(I)+"\n")
#####################################################################
# *** Method 3: Simpson's rule ***

I = 0
for i in range(len(fx)-1):
    I_n = ((x[i+1]-x[i])/2)/3 * (f(x[i+1])+4*f(1/2*(x[i+1]+x[i]))+f(x[i]))
    I += I_n

print("3) Simpson's: "+str(I)+"\n")
#####################################################################
# *** Method 4: Simpson's composite rule ***

I = 0
for i in range(len(fx)):

    if i == 0 or i == len(fx)-1:
        c = 1
    else:
        if i % 2 != 0: # i is odd
            c = 4
        else: # i is even
            c = 2
    
    I += c * f(x[i])

I = (x[1]-x[0])/3 * I
print("4) Composite Simpson's: "+str(I)+"\n")
