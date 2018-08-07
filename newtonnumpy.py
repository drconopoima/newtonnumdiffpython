"""
Newton-Raphson Method 1-variable

This script finds the root of a function by taking the function and its
derivative, one seed value to start the iterations and one relative tolerance
between successive values
Python with numpy extension modules

Created on Tue Sep 22 15:01:36 2015

@author: Luis Jesus Diaz
"""

import sys, numpy as np, matplotlib.pyplot as plt

# Solucion al sistema univariable: [0.3, 3.7]

def func(x):
    """
    Objective function whose roots are the ones to find.
    """
    return (x - 1) ** 2 - 2 * x + 0.11


def dfuncdx(x, **kwargs):
    """
    Derivative of the objective function.
    """
    return 2 * x - 4


def numericdiff5(x, func, functol=1e-8):
    return (-func(x+2*functol)+8*func(x+functol)-8*func(x-functol)+func(x-2*functol))/(12*functol)


def newton(func, seed, dfuncdx='none', tol=1e-6, functol=1e-8, damping=1, maxiter=1000, moreoutput=0):
    """
    Function that finds the roots of func using Newton-Raphson's method
    By default, the derivative is found using 3-point numeric differentiation
    but user can input a function in argument dfuncdx to use it to evaluate
    exact derivative
    Optional input arguments are tol=1e-6, functol=1e-8, damping=1, maxiter=1000
    Output arguments are:
        - x1: the final roots found
        - dev: the relative errors of the roots between last 2 iterations
        - backupx0: for graphing purposes it saves root trials each iteration
        - backupdev: errors in every iteration
        - backupdx: correction factors every iteration
        - itera: number of iterations
    """
    itera = 0
    seed = np.atleast_1d(seed)
    seed = seed.astype(np.float64)
    x0 = seed  # initial guess
    n = len(seed)  # number of initial guesses
    functol = np.atleast_1d(functol)
    functol = functol.astype(np.float64)
    backupx0 = np.empty(shape=[0, n])
    if moreoutput == 1:
        backupdev = np.empty(shape=[0, n])
        backupdx = backupdev.copy()
        backupf0 = backupdev.copy()
        backupdf0 = backupdev.copy()
    dev = np.array([sys.float_info.max] * n)  # initial error assumed biggest float in python
    # if no derivative was passed default to 3-point numerical differentiation
    if dfuncdx == 'none':
        def dfuncdx(x, func=func, functol=functol):
            return ((func(x+functol)-func(x-functol))/(2*functol))
    while np.any(dev > tol) and (itera < maxiter):
        itera += 1
        # new set of variables in the new iteration
        f0 = func(x0).tolist() # valor de la funcion esta iteracion
        df0 = dfuncdx(x0, func=func, functol=functol).tolist() # valor de la derivada esta iteracion
        if moreoutput == 1:
            backupf0 = np.append(backupf0, f0)
            backupdf0 = np.append(backupdf0, df0)
        dx = []
        for iterfor0 in range(n):
            try:
                dx.append(damping * f0.pop() / df0.pop())
            except:
                dx.append(0.)
        dx = np.array(dx)
        x1 = (x0 - dx)
        if min(abs(x0)) > min([min(functol), 1e-8]) * 1e-2:
            dev = abs(x0 - x1) / abs(x0)  # error relativo si no es muy cercano a cero
        else:
            dev = abs(x0 - x1)              # error absoluto si es cercano a cero
        if moreoutput == 1:
            backupdev = np.append(backupdev, dev)
            backupdx = np.append(backupdx, dx)
        backupx0 = np.append(backupx0, x0)
        x0 = x1
    backupx0 = np.append(backupx0, x0)
    backupx0 = backupx0.reshape([itera + 1, n])
    if moreoutput == 1: 
        backupdev = backupdev.reshape([itera, n])
        backupdx = backupdx.reshape([itera, n])
        backupf0 = backupf0.reshape([itera, n])
        backupdf0 = backupdf0.reshape([itera, n])
    if moreoutput == 1:
        return x1, itera, dev, backupx0, backupdev, backupdx, backupf0, backupdf0
    else:
        return x1, itera, dev, backupx0


#Si se desea usar derivada analitica
funcion = func
derivada = dfuncdx
x1, iter1, desv1, histx1 = newton(func=func, dfuncdx=derivada, seed=[0, 5])

# Si se desea usar derivada numerica de 5 puntos
derivada = numericdiff5
x2, iter2, desv2, histx2 = newton(func=func, dfuncdx=derivada, seed=[0, 5])

# Si se desea usar derivada por defecto de 3 puntos centrada lineal
x3, iter3, desv3, histx3 = newton(func=func, seed=[0, 5])


# Otros ejemplos
def funcprueba1(x):
    return -x ** 2 - 5 * x - 3 + np.exp(x)

# resuelto con derivada numerica de 5 puntos
x4, iter4, desv4, histx4 = newton(func=funcprueba1, seed=[0, 5], dfuncdx=numericdiff5)


def funcprueba2(x):
    return -1 * (1 / 5600) * (x + 5) ** 2 * (x + 1) * (x - 4) ** 3 * (x - 7)


# sus raices son -5, -1, +4 y +7, se le daran equiespaciados 5 semillas entre -6 y +8
x5, iter5, desv5, histx5 = newton(func=funcprueba2, seed=np.linspace(-6,8,5), dfuncdx=numericdiff5)


def funcobjetivo(x):
    return (x-5)**2


# Para obtener datos adicionales de cada iteracion se usa moreoutput=1
x6, iter6, desv6, histx6, backupdesv6, backupdx6, backupf0, backupdf0 = newton(func=funcobjetivo, seed=[1.], dfuncdx=numericdiff5, moreoutput=1)

# Para resolver comparable al algoritmo matlab, se fijara las mismas tolerancias
# tanto para el tamano del paso: functol = 1e-6 como la tolerancia relativa
# tol = 1e-6

x7, iter7, desv7, histx7 = newton(func=funcobjetivo, seed=[1.], dfuncdx=numericdiff5)


# Para graficar el resultado comparandolo al de matlab

arg = np.linspace(-6.2, 6.2, 100)
fun = funcobjetivo(arg)
itera = np.linspace(0,iter7, iter7+1)
cero = np.zeros(100)
f, axarr = plt.subplots(2)
ax1 = plt.subplot(131)
ax1.plot(itera, histx7)
ax1.set_ylim([.999,5.001])
ax2 = plt.subplot(132)
ax2.plot(arg, fun, arg, cero)
ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-10, 130)


