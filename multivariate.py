"""
Multivariate Newton-Raphson Method

This script finds the roots of defined functions by taking them and their
jacobian matrix, seed values to start the iterations and the relative tolerance
between successive values
To estimate the jacobian matrix uses 3-point numeric differentiation
multivariate.py requires numpy package

Created on Tue Sep 22 20:30:59 2015

@author: Luis Jesus Diaz
"""

# Sistema no lineal de ecuaciones
# {
# (x1)^2 - 10 * x1 + (x2)^2 + 8 = 0
# x1 * x2^2 + x1  - 10 * x2 + 8 = 0
# }
# soluciones al sistema multivariable: x=[1,1] o x=[2.19343942,  3.02046647]
# tiene mas de 1 por no tratarse de un sistema lineal

import sys, numpy as np, matplotlib.pyplot as plt

def func1(x):
    """
    Objective function 1 whose roots are the ones to find.
    """
    x = x.ravel()
    return x[0] ** 2 - 10 * x[0] + x[1] ** 2 + 8


def func2(x):
    """
    Objective function 2 whose roots are the ones to find.
    """
    x = x.ravel()
    return x[0] * x[1] ** 2 + x[0]  - 10 * x[1] + 8

# De ser necesario, se incrementa el orden del sistema de ecuaciones definiendo mas funciones
# ==============================================================================
#  def func3(x):
#      """
#      Objective function 3 whose roots are the ones to find.
#      """
#      x = x.ravel()
#      return (x[0]+2)**3 - 8*x[2] - x[0]*np.exp(x[1]) + 4.3890560989306504
# ==============================================================================

func = [func1, func2]


def numericdiff3(x, func=func, functol=1e-8):
    """
    x preferiblemente debe ser una lista o un iterable
    """
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        x = [x]
    x = np.atleast_1d(x)
    x = x.astype(np.float64)
    n = len(x)
    m = len(func)
    salida = np.zeros([m, n])
    for iterfor0 in range(m):
        for iterfor1 in range(n):
            xtrial0 = x.copy()
            xtrial0[iterfor1] -= functol
            xtrial1 = x.copy()
            xtrial1[iterfor1] += functol
            salida[iterfor0, iterfor1] = (func[iterfor0](xtrial1) - func[iterfor0](xtrial0)) / (2.0 * functol)
    return salida


def numericdiff5(x, func=func, functol=1e-8):
    """
    x preferiblemente debe ser una lista o un iterable
        
    functol es el tamano del paso del metodo numerico        
    """
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        x = [x]
    x = np.atleast_1d(x)
    x = x.astype(np.float64)
    n = len(x)
    m = len(func)
    salida = np.zeros([m, n])
    for iterfor0 in range(m):
        for iterfor1 in range(n):
            xtrial0 = x.copy()
            xtrial0[iterfor1] -= 2*functol
            xtrial1 = x.copy()
            xtrial1[iterfor1] -= functol
            xtrial3 = x.copy()
            xtrial3[iterfor1] += functol
            xtrial4 = x.copy()
            xtrial4[iterfor1] += 2*functol
            salida[iterfor0, iterfor1] = (-func[iterfor0](xtrial4)+8*func[iterfor0](xtrial3)-8*func[iterfor0](xtrial1)+func[iterfor0](xtrial0))/(12*functol)
    return salida


def newton(seed, func=func, dfuncdx=numericdiff3, tol=1e-6, functol=1e-8, damping=1, maxiter=1000, moreoutput=0):
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
    x0 = seed # initial guess
    x0 = x0.astype(np.float64)
    n = len(seed) # number of roots
    m = len(func)
    dev = np.array([sys.float_info.max] * n, dtype=np.float64)
    backupx0 = np.empty([0, n])
    if moreoutput == 1:
        backupdx = np.empty([0, n])
        backupdev = backupdx.copy()
        backupf0 = np.empty([0, m])
        backupdf0 = backupdx.copy()
    while np.any(dev > tol) and (itera < maxiter):
        itera += 1
        funvectorial = np.zeros(n)
        for iterfor2 in range(m):
            funvectorial[iterfor2] = -func[iterfor2](x0)
        jacob = dfuncdx(x0, func, functol)
        if moreoutput == 1:
            backupf0 = np.append(backupf0, funvectorial)
            backupdf0 = np.append(backupdf0, jacob)
        dx = damping * np.linalg.solve(jacob, funvectorial)
        x1 = x0 + dx
        dev = abs(x0 - x1) / abs(x0)  # error
        backupx0 = np.append(backupx0, x0)
        if moreoutput == 1:
            backupdev = np.append(backupdev, dev)
            backupdx = np.append(backupdx, dx)
        x0 = x1
    backupx0 = np.append(backupx0, x0)
    backupx0 = backupx0.reshape([itera + 1, n])
    if moreoutput == 1:
        backupdev = backupdev.reshape([itera, n])
        backupdx = backupdx.reshape([itera, n])
        backupf0 = backupf0.reshape([itera, m])   
        backupdf0 = backupdf0.reshape([(itera) * m, n])
    if moreoutput == 1:
        return x1, itera, dev, backupx0, backupdev, backupdx, backupf0, backupdf0
    else:
        return x1, itera, dev, backupx0

x11, iter1, dev1, backupx01 = newton([0, 0], func=func, dfuncdx=numericdiff3)

x12, iter2, dev2, backupx02 = newton([1.5, 2.5], func=func, dfuncdx=numericdiff5)

x13, iter3, dev3, backupx03, backupdev3, backupdx3, backupf0, backupdf03 = newton([1., 6.5], func=func, dfuncdx=numericdiff5, moreoutput=1)

