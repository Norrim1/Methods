import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

a = 1.5
b = 1.8

def w(x):
    return -9 * x + 14.1

def w_prime(x):
    return -9

def w_double_prime(x):
    return 0

print(f"w(a) = {w(a)}")
print(f"2*w(b) - 0.8*w'(b) = {2*w(b) - 0.8*w_prime(b)}")
b_val = 1.8
a_val = 1.5

L = b - a

def t(x):
    return (x - a) / L

def psi(x):
    return (x - 1.5) * (x - 3.0)

def psi_prime(x):
    return 2*x - 4.5

def psi_double_prime(x):
    return 2.0

def phi0(x):
    return psi(x)

def phi1(x):
    return (x - a) * psi(x)

def phi2(x):
    return (x - a)**2 * psi(x)

def phi0_prime(x):
    return psi_prime(x)

def phi0_double(x):
    return psi_double_prime(x)

def phi1_prime(x):
    return psi(x) + (x - a) * psi_prime(x)

def phi1_double(x):
    return 2 * psi_prime(x) + (x - a) * psi_double_prime(x)


def phi2_prime(x):
    return 2*(x-a)*psi(x) + (x-a)**2 * psi_prime(x)

def phi2_double(x):
    return 2*psi(x) + 4*(x-a)*psi_prime(x) + (x-a)**2 * psi_double_prime(x)

phis = [phi0, phi1, phi2]

phis_prime = [phi0_prime,phi1_prime,phi2_prime]

phis_double = [phi0_double,phi1_double,phi2_double]

print(f"psi(a) = {psi(a)}")
print(f"2*psi(b) - 0.8*psi'(b) = {2*psi(b) - 0.8*psi_prime(b)}")

def L_operator(u_func, u_prime_func, u_double_func, x):
    return u_double_func(x) + 0.6 * x * u_prime_func(x) - 2 * u_func(x)

def Rw(x):
    return L_operator(w, w_prime, w_double_prime, x) - 1

def Rpsi(x):
    return L_operator(psi, psi_prime, psi_double_prime, x)

def integrand_num(x):
    return Rw(x) * psi(x)

def integrand_den(x):
    return Rpsi(x) * psi(x)

A_gal = np.zeros((3, 3))
B_gal = np.zeros(3)

for j in range(3):

    def rhs_integrand(x):
        return Rw(x) * phis[j](x)

    B_gal[j] = -quad(rhs_integrand, a, b)[0]

    for i in range(3):

        def lhs_integrand(x):
            return (L_operator(phis[i],phis_prime[i],phis_double[i],x)* phis[j](x))

        A_gal[j, i] = quad(lhs_integrand, a, b)[0]

c = np.linalg.solve(A_gal, B_gal)

print("Коэффициенты Галеркина:")
print(f"c0 = {c[0]}")
print(f"c1 = {c[1]}")
print(f"c2 = {c[2]}")

def y_galerkin(x):
    return (w(x)+ c[0] * phi0(x)+ c[1] * phi1(x)+ c[2] * phi2(x))

def solve_bvp_fdm(n):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = 1.0; D[0] = 0.6
    
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
        alpha[i] = -C[i] / denom
        beta[i] = (D[i] - A[i] * beta[i - 1]) / denom

    y = np.zeros(n)
    y[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]
        
    return x, y

x_fdm_coarse, y_fdm_coarse = solve_bvp_fdm(11)
x_fdm_fine, y_fdm_fine = solve_bvp_fdm(101)

x_gal = np.linspace(a, b, 100)
y_gal_vals = [y_galerkin(val) for val in x_gal]

plt.figure(figsize=(12, 7))

plt.plot(x_gal, y_gal_vals, 'g-', linewidth=2, label='Метод Галеркина (n=2)')
plt.plot(x_fdm_fine, y_fdm_fine, 'b--', linewidth=2, label='MKP (N=101)')
plt.plot(x_fdm_coarse, y_fdm_coarse, 'ro', markersize=5, label='MKP (N=11)')

plt.title('Сравнение метода Галеркина и метода конечных разностей')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

y_gal_on_fdm_nodes = [y_galerkin(val) for val in x_fdm_fine]
error_galerkin = np.abs(np.array(y_gal_on_fdm_nodes) - y_fdm_fine)

plt.figure(figsize=(12, 5))
plt.plot(x_fdm_fine, error_galerkin, 'm-', linewidth=2)
plt.title('Погрешность метода Галеркина относительно решения MKP (N=101)')
plt.xlabel('x')
plt.ylabel('Abs Error')
plt.grid(True)
plt.show()

print(f"Максимальная погрешность Галеркина (относительно MKP N=101): {np.max(error_galerkin):.6e}")