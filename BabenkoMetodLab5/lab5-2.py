import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x, t):
    return x + 1 + t**2 * x


def thomas_algorithm(a, b, c, d):
    n = len(d)

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denom if i < n-1 else 0.0
        beta[i] = (d[i] - a[i] * beta[i-1]) / denom

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x

def solve_heat_equation_implicit(nx, T=1.0):

    x = np.linspace(0, 1, nx)
    h = 1.0 / (nx - 1)
    tau = h
    nt = int(T / tau) + 1
    tau = T / (nt - 1)
    t = np.linspace(0, T, nt)

    U = np.zeros((nt, nx))
    U[0, :] = x + 1

    for n in range(nt - 1):

        left = 1.0
        right = 2.0 + t[n+1]**2

        m = nx - 2

        a = np.zeros(m)
        b = np.zeros(m)
        c = np.zeros(m)
        d = np.zeros(m)

        A = -tau * (1/h**2 - 1/h)
        B = 1 + 2*tau/h**2
        C = -tau * (1/h**2 + 1/h)

        for i in range(m):

            a[i] = A
            b[i] = B
            c[i] = C

            d[i] = U[n, i+1]

        d[0] -= A * left
        d[-1] -= C * right

        u_inner = thomas_algorithm(a, b, c, d)

        U[n+1, 0] = left
        U[n+1, -1] = right
        U[n+1, 1:-1] = u_inner

    return x, t, U, h, tau

nx_values = [11, 51, 101]

results = {}
for nx in nx_values:

    x, t, U, h, tau = solve_heat_equation_implicit(nx)

    results[nx] = {"x": x,"t": t,"U": U,"h": h,"tau": tau}

    error = np.max(np.abs(U[-1] - exact_solution(x, 1.0)))

    print(f"N = {nx}")
    print(f"h = {h}")
    print(f"tau = {tau}")
    print(f"max error = {error}")

plt.figure(figsize=(10,6))

for nx in nx_values:

    x = results[nx]["x"]
    U = results[nx]["U"]

    plt.plot(x,U[-1],label=f"N={nx}")

x_exact = np.linspace(0,1,500)

plt.plot(x_exact,exact_solution(x_exact,1),'k--',linewidth=2,label='Точное решение')

plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.title("Неявная схема")
plt.legend()
plt.grid()
plt.show()

errors = []
h_values = []

for nx in nx_values:
    x = results[nx]["x"]
    U = results[nx]["U"]
    error = np.max(np.abs(U[-1] - exact_solution(x,1)))
    errors.append(error)
    h_values.append(results[nx]["h"])

plt.figure(figsize=(8,6))

plt.loglog(h_values,errors,'o-',linewidth=2,label='Ошибка')
plt.loglog(h_values,[h for h in h_values],'--',label='O(h)')
plt.loglog(h_values,[h**2 for h in h_values],'--',label='O(h²)')

plt.xlabel("h")
plt.ylabel("max error")
plt.title("Сходимость")
plt.grid(True, which="both")
plt.legend()
plt.show()