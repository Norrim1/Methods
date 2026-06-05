import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(n):
    a = 1.5
    b = 1.8
    
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = 1.0
    C[0] = 0.0
    D[0] = 0.6

    for i in range(1, n - 1):
        A[i] = 1 / h**2 - 0.6 * x[i] / (2 * h)
        B[i] = -2 / h**2 - 2
        C[i] = 1 / h**2 + 0.6 * x[i] / (2 * h)
        D[i] = 1.0
    
    xi = x[n-1]
    K = 2.5 * h
    
    A[n-1] = 2 / h**2
    B[n-1] = (-2 / h**2) + (2 * K / h**2) + (1.5 * xi) - 2
    D[n-1] = 1 + (3 * K / h**2) + (4.5 * xi)
    C[n-1] = 0.0

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -C[0] / B[0]
    beta[0] = D[0] / B[0]

    for i in range(1, n):
        denom = B[i] + A[i] * alpha[i - 1]
        if abs(denom) < 1e-12:
            raise ZeroDivisionError(f"Деление на ноль на шаге {i}")
        
        alpha[i] = -C[i] / denom
        beta[i] = (D[i] - A[i] * beta[i - 1]) / denom

    y = np.zeros(n)
    y[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]

    return x, y

grids = [11, 51, 101]

plt.figure(figsize=(10, 6))

for n in grids:
    x, y = solve_bvp(n)
    plt.plot(x, y, label=f'N = {n}', linewidth=2 if n == 101 else 1)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение краевой задачи методом конечных разностей')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()