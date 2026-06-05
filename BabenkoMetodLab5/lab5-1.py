import numpy as np
import matplotlib.pyplot as plt

def f_func():
    return 0.0

def exact_solution(x, t):
    return x + 1 + t**2 * x

def solve_heat_equation_explicit(nx, tau=None, T=1.0):
    x = np.linspace(0, 1, nx)
    h = 1.0 / (nx - 1)

    if tau is None:
        tau = 0.9 * h**2 / (2 + 2*h)
    
    nt = int(T / tau) + 1
    tau = T / (nt - 1)
    t = np.linspace(0, T, nt)

    U = np.zeros((nt, nx))
    U[0, :] = x + 1
    U[:, 0] = 1.0
    U[:, -1] = 2 + t**2

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            d2U = (U[n, i+1] - 2*U[n, i] + U[n, i-1]) / h**2
            dU = (U[n, i+1] - U[n, i-1]) / (2*h)
            U[n+1, i] = U[n, i] + tau * (d2U + 2*dU + f_func())
    
    return x, t, U, tau

def analyze_stability(nx):
    h = 1.0 / (nx - 1)
    tau_max = h**2 / (2 + 2*h)
    cfl_diff = tau_max / h**2
    cfl_conv = 2 * tau_max / h
    
    return h, tau_max, cfl_diff, cfl_conv

nx_values = [11, 51, 101]
results = {}

for nx in nx_values:
    x, t, U, tau = solve_heat_equation_explicit(nx)
    h, tau_max, cfl_diff, cfl_conv = analyze_stability(nx)
    
    results[nx] = {'x': x, 't': t, 'U': U, 'tau': tau, 'h': h}
    
    print(f" Сетка c {nx} узлами:")
    print(f"  Шаг по x: h = {h}")
    print(f"  Шаг по t: τ = {tau}")
    print(f"  Максимальный τ для устойчивости: {tau_max}")
    print(f"  Число CFL (диффузия): {cfl_diff}")
    print(f"  Число CFL (конвекция): {cfl_conv}")
    print(f"  Условие устойчивости: {'ВЫПОЛНЯЕТСЯ' if tau <= tau_max else 'НАРУШАЕТСЯ'}")

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(10, 6))
for nx in nx_values:
    x = results[nx]['x']
    U = results[nx]['U']
    ax.plot(x, U[-1, :], 'o-', label=f'N={nx}', markersize=4, alpha=0.7)

x_exact = np.linspace(0, 1, 100)
t_val = 1.0
U_exact = exact_solution(x_exact, t_val)
ax.plot(x_exact, U_exact, 'k--', linewidth=2, label='Точное решение')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('U(x, T=1)', fontsize=12)
ax.set_title('Решение уравнения теплопроводности в момент T=1 для разных разностных сеток', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

nx_plot = 51
x = results[nx_plot]['x']
t = results[nx_plot]['t']
U = results[nx_plot]['U']

X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, U, cmap='viridis', alpha=0.9)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('t', fontsize=12)
ax.set_title('Решение уравнения теплопроводности (явная схема, N=51)', fontsize=14)
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

errors = []
h_values = []
for nx in nx_values:
    x = results[nx]['x']
    U = results[nx]['U']
    t = results[nx]['t']
    h = results[nx]['h']
    
    U_exact_final = exact_solution(x, t[-1])
    error = np.max(np.abs(U[-1, :] - U_exact_final))
    errors.append(error)
    h_values.append(h)

ax.loglog(h_values, errors, 'o-', linewidth=2, markersize=10, label='Численная погрешность')
ax.loglog(h_values, [0.1*h**2 for h in h_values], '--', label='O(h²)')
ax.loglog(h_values, [h for h in h_values], '--', label='O(h)')

ax.set_xlabel('Шаг сетки h', fontsize=12)
ax.set_ylabel('Максимальная погрешность', fontsize=12)
ax.set_title('Исследование сходимости схемы', fontsize=14)
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

nx_plot = 51
x = results[nx_plot]['x']
t = results[nx_plot]['t']
U = results[nx_plot]['U']

time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, len(t)-1]
time_labels = [f't={t[i]:.2f}' for i in time_indices]

for idx, label in zip(time_indices, time_labels):
    ax.plot(x, U[idx, :], linewidth=2, label=label)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('U(x,t)', fontsize=12)
ax.set_title('Эволюция решения во времени (N=51)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()