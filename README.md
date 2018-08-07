# newtonnumdiffpython
Newton-Raphson Method 1-variable  This script finds the root of a function by taking the function and its derivative, one seed value to start the iterations and one relative tolerance between successive values Python with numpy extension modules

This methods depends on packages NumPy and MatplotLib.

Examples

# Getting Started

To run use the iPython prompt or your favorite IDE

Change to the directory where you saved the files and run

```
%run newtonpy.py
```

To measure performance and optimize, you can take the time of the algorithm using:

```
timeit newton(func=functionalobjective, dfuncdx=numericdiff5, seed=[1.], tol=1e-6, functol=1e-6, )
```

There tol is the relative root error found (although I use it absolutely if the root found is close to zero from -0.1 to 0.1)
functol is the size of the absolute step of the numerical derivation, not used if the derivative supplied is analytical

numericdiff5 is the centered 5-point numerical derivative

You can also solve several seeds simultaneously, for example the second example you sent by mail that has 4 roots, you can find them all by making a linspace of 5 elements between -6 and 8

```
x, iter, desv, histx = newton(func=functest2, seed=np.linspace(-6,8,5), dfuncdx=numericdiff5)
```

The multivariable version is very similar, the biggest difference is that it does not accept or solve several seeds simultaneously, the only seed entered must be the size of the system variables. It can be used by running the script

```
%run multivariate.py
```

Can be made to calculate the matrix of derivatives with 5-point numerical differential


## Example

```
func1 = (x1)^2 - 10 * x1 + (x2)^2 + 8 = 0
func2 = x1 * x2^2 + x1 - 10 * x2 + 8 = 0
```

Function matrix is defined with a list

```
func =[func1, func2]
```

Then, the resolution of the multivariate Newton Raphson differentiation can be done with the following line to run the script

```
x, iter, dev, histx = newton([1.5, 2.5], func=func, dfuncdx=numericdiff5)
```
