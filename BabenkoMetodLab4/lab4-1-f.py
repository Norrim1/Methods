import numpy as np
import pandas as pd

n = 11
a = 0
b = 1
h = (b - a) / (n - 1)

x = np.linspace(a, b, n)

A = np.zeros(n)
B = np.zeros(n)
C = np.zeros(n)
D = np.zeros(n)

for i in range(1, n - 1):
    A[i] = 1 / h**2 - x[i] / (2 * h)
    B[i] = -2 / h**2 - 2
    C[i] = 1 / h**2 + x[i] / (2 * h)
    D[i] = 2

y = np.zeros(n)
y[0] = 0 
y[n - 1] = 1

m = n - 2 

A_in = np.zeros(m)
B_in = np.zeros(m)
C_in = np.zeros(m)
D_in = np.zeros(m)

for j in range(m):
    i = j + 1
    A_in[j] = A[i]
    B_in[j] = B[i]
    C_in[j] = C[i]
    D_in[j] = D[i]

D_in[0] -= A_in[0] * y[0]
D_in[m - 1] -= C_in[m - 1] * y[n - 1]

alpha = np.zeros(m)
beta = np.zeros(m)

alpha[0] = -C_in[0] / B_in[0]
beta[0] = D_in[0] / B_in[0]

for j in range(1, m):
    denom = B_in[j] + A_in[j] * alpha[j - 1]
    if abs(denom) < 1e-12:
        raise ZeroDivisionError(f"Деление на ноль на шаге {j}")
    alpha[j] = -C_in[j] / denom
    beta[j] = (D_in[j] - A_in[j] * beta[j - 1]) / denom

y[n - 2] = beta[m - 1]

y_internal = np.zeros(m)
y_internal[m - 1] = beta[m - 1]
C_in[m - 1] = 0 
alpha = np.zeros(m)
beta = np.zeros(m)

alpha[0] = -C_in[0] / B_in[0]
beta[0] = D_in[0] / B_in[0]

for j in range(1, m):
    denom = B_in[j] + A_in[j] * alpha[j - 1]
    alpha[j] = -C_in[j] / denom
    beta[j] = (D_in[j] - A_in[j] * beta[j - 1]) / denom

y_internal[m - 1] = beta[m - 1]
for j in range(m - 2, -1, -1):
    y_internal[j] = alpha[j] * y_internal[j + 1] + beta[j]

y[1:n - 1] = y_internal

y_exact = x**2

error = np.abs(y - y_exact)


df = pd.DataFrame({'x': x,'y_num': y,'y_exact': y_exact,'error': error})

print(df.to_string(index=False))
print(f"Максимальная погрешность: {np.max(error):.6e}")