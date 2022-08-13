import numpy as np
from functions import gauss_jacobi as gs


A = np.array([[10, 2, 1],[1, 5, 1],[2, 3, 10]], dtype = float)
b = np.array([7, -8, 6], dtype = float)
guess = np.array([0.7, -1.6, 0.6], dtype = float)

sol = gs.jacobi(A,b,E=0.001,N=2,x=guess)

print ("A:")
print(A)

print ("b:")
print(b)

print ("x:")
print(sol)