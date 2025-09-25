#2a. Sistema determinista
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from numba import njit

# Datos iniciales
A = 1000  
B = 20     
# Tiempos de vida media en días
t_U = 23.4 / (24 * 60)  # Pasamos de minutos a días
t_Np = 2.36

# Constantes de decaimiento lambda calculadas con la formula lambda_U = np.log(2) / t_U
lambda_Np = np.log(2) / t_Np
lambda_U = np.log(2) / t_U

# Ecuaciones diferenciales dadas en el problema
def ecuaciones_diferenciales(t, y):
    U, Np, Pu = y
    dU_dt = A - lambda_U * U
    dNp_dt = lambda_U * U - lambda_Np * Np
    dPu_dt = lambda_Np * Np - B * Pu
    return [dU_dt, dNp_dt, dPu_dt]

# Condiciones iniciales
U0, Np0, Pu0 = 10, 10, 10
tiempo = 30 # Tiempo de simulación de 30 días

# Resolvemos el sistema 
sol = solve_ivp(ecuaciones_diferenciales, [0, tiempo], [U0, Np0, Pu0], t_eval=np.linspace(0, tiempo, 300))

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Uranio-239 (U)')
plt.plot(sol.t, sol.y[1], label='Neptunio-239 (Np)')
plt.plot(sol.t, sol.y[2], label='Plutonio-239 (Pu)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Cantidad de material')
plt.yscale('log')
plt.legend()
plt.title('Evolución de las cantidades de U, Np y Pu en 30 días')
plt.grid()
plt.savefig("Taller 5/2.a.png")

# Verificamos estabilidad
# Función para detectar estabilidad
def tiempo_estabilidad(t, y, dydt, tol=0.01):
    for i in range(len(t)):
        if abs(dydt[i]) < tol:
            return t[i]
    return None

# Calcular derivadas en los puntos

derivs = []  
for i in range(len(sol.t)):          # recorremos cada tiempo
    t_actual = sol.t[i]              # tiempo en el paso i
    y_actual = sol.y[:, i]           # valores (U, Np, Pu) en ese paso
    dydt = ecuaciones_diferenciales(t_actual, y_actual)  # calculamos derivadas
    derivs.append(dydt)              # guardamos

# Convertimos la lista a un arreglo numpy (para facilidad)
derivs = np.array(derivs)
print(derivs.shape)
# Evaluación de cada isótopo
isotopos = ["U (Uranio-239)", "Np (Neptunio-239)", "Pu (Plutonio-239)"]
for i, iso in enumerate(isotopos):
    t_est = tiempo_estabilidad(sol.t, sol.y[i], derivs[:, i]) #Si la dereivada es menor a tol, se considera estable
    if t_est is not None:
        print(f"{iso}: Llega a estado estable en t ≈ {t_est:.3f} días, valor ≈ {sol.y[i,-1]:.2f}")
    else:
        print(f"{iso}: NO alcanza estado estable en 30 días, valor final ≈ {sol.y[i,-1]:.2f}")
# 2b. Ecuación diferencial estocástica (Runge-Kutta estocástico de orden 2)

def sde_rk2(A, B, lambda_U, lambda_Np, tiempo, dt=0.01, U0=10, Np0=10, Pu0=10):
    pasos = int(tiempo/dt)
    t_vals = np.linspace(0, tiempo, pasos)
    
    # Arreglos para guardar resultados
    U = np.zeros(pasos); Np = np.zeros(pasos); Pu = np.zeros(pasos)
    U[0], Np[0], Pu[0] = U0, Np0, Pu0

    for i in range(1, pasos):
        # ---- Evolución de U con ruido estocástico ----
        muU = A - lambda_U * U[i-1]
        sigmaU = np.sqrt(A + lambda_U * U[i-1])
        W = np.random.normal(0,1)
        S = np.random.choice([-1,1])

        K1 = dt*muU + (W+S)*np.sqrt(dt)*sigmaU
        K2 = dt*(A - lambda_U*(U[i-1]+K1)) + (W+S)*np.sqrt(dt)*np.sqrt(A + lambda_U*(U[i-1]+K1))

        U[i] = U[i-1] + 0.5*(K1+K2)

        # ---- Determinista para Np y Pu ----
        dNp = (lambda_U * U[i-1] - lambda_Np * Np[i-1]) * dt
        dPu = (lambda_Np * Np[i-1] - B * Pu[i-1]) * dt

        Np[i] = Np[i-1] + dNp
        Pu[i] = Pu[i-1] + dPu

    return t_vals, U, Np, Pu


# Graficar 5 trayectorias + determinista en subplots
num_trayectorias = 5
fig, axes = plt.subplots(3, 1, figsize=(10,12), sharex=True)

# Nombres de isótopos
isotopos = ["Uranio-239 (U)", "Neptunio-239 (Np)", "Plutonio-239 (Pu)"]

# Guardamos trayectorias estocásticas
datos = []
for i in range(num_trayectorias):
    t_vals, U_vals, Np_vals, Pu_vals = sde_rk2(A, B, lambda_U, lambda_Np, tiempo)
    datos.append((t_vals, U_vals, Np_vals, Pu_vals))

# Dibujar en subplots
for idx, (ax, iso) in enumerate(zip(axes, isotopos)):
    # Solución determinista ya calculada en 2a
    ax.plot(sol.t, sol.y[idx], "k--", label="Determinista")
    
    # Trayectorias estocásticas
    for i, (t_vals, U_vals, Np_vals, Pu_vals) in enumerate(datos):
        if idx == 0:
            ax.plot(t_vals, U_vals, alpha=0.7, label=f"Trayectoria {i+1}")
        elif idx == 1:
            ax.plot(t_vals, Np_vals, alpha=0.7, label=f"Trayectoria {i+1}")
        else:
            ax.plot(t_vals, Pu_vals, alpha=0.7, label=f"Trayectoria {i+1}")
    
    ax.set_ylabel("Cantidad")
    ax.set_title(iso)
    ax.grid(True)
    if idx == 2:
        ax.set_xlabel("Tiempo (días)")
    ax.legend()

plt.tight_layout()
plt.savefig("Taller 5/2.b.png")
#Punto 2c.
import random
R = np.array([
    [1, 0, 0],   # creación de U
    [-1, 1, 0],  # U -> Np
    [0, -1, 1],  # Np -> Pu
    [0, 0, -1]   # decaimiento de Pu
])

@njit
def simulacion_gillespie(tiempo_simulacion, A, lambda_U, lambda_Np, B, 
                         U0=10, Np0=10, Pu0=10, max_pasos=200000):
    # Estado inicial
    t_old = 0.0
    Y_old = np.array([U0, Np0, Pu0])  # vector [U, Np, Pu]

    # Vectores para guardar resultados
    tiempos = np.zeros(max_pasos)
    valores_U = np.zeros(max_pasos)
    valores_Np = np.zeros(max_pasos)
    valores_Pu = np.zeros(max_pasos)

    indice = 0

    while t_old < tiempo_simulacion and indice < max_pasos - 1:
        # 1. Calcular tasas
        tasa_creacion = A
        tasa_U = Y_old[0] * lambda_U
        tasa_Np = Y_old[1] * lambda_Np
        tasa_Pu = Y_old[2] * B
        Tasas = np.array([tasa_creacion, tasa_U, tasa_Np, tasa_Pu])

        if Tasas.sum() == 0:
            break

        # 2. Tiempo hasta la siguiente reacción
        tau = np.random.exponential(1 / Tasas.sum())

        # 3. Elegir reacción
        probabilidades = Tasas / Tasas.sum()
        r_index = np.searchsorted(np.cumsum(probabilidades), np.random.rand())  # elige la reacción sorteando un número aleatorio dentro de los intervalos de probabilidad acumulada
        r = R[r_index]  # obtiene el vector de cambios (ΔU, ΔNp, ΔPu) correspondiente a esa reacción


        # 4. Actualizar estado
        Y_new = Y_old + r

        # 5. Evolucionar tiempo
        t_new = t_old + tau

        # Guardar resultados
        tiempos[indice] = t_new
        valores_U[indice] = Y_new[0]
        valores_Np[indice] = Y_new[1]
        valores_Pu[indice] = Y_new[2]

        # Avanzar para siguiente paso
        Y_old = Y_new
        t_old = t_new
        indice += 1

    return tiempos[:indice], valores_U[:indice], valores_Np[:indice], valores_Pu[:indice]
# Parámetros 
tiempo_simulacion = 30
num_trayectorias = 5

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
variables = ["Uranio-239 (U)", "Neptunio-239 (Np)", "Plutonio-239 (Pu)"]

# Solución determinista
axs[0].plot(sol.t, sol.y[0], 'k-', lw=2, label="Determinista")
axs[1].plot(sol.t, sol.y[1], 'k-', lw=2, label="Determinista")
axs[2].plot(sol.t, sol.y[2], 'k-', lw=2, label="Determinista")

# Trayectorias estocásticas
for i in range(num_trayectorias):
    tiempos, U, Np, Pu = simulacion_gillespie(tiempo_simulacion, A, lambda_U, lambda_Np, B)
    axs[0].step(tiempos, U, where="post", alpha=0.7)
    axs[1].step(tiempos, Np, where="post", alpha=0.7)
    axs[2].step(tiempos, Pu, where="post", alpha=0.7)

# Ajustes
for i, ax in enumerate(axs):
    ax.set_ylabel(variables[i])
    ax.legend()
    ax.grid()

axs[2].set_xlabel("Tiempo (días)")
plt.suptitle("2.c. Gillespie vs Solución determinista")
plt.tight_layout()
plt.savefig("Taller 5/2.c.png")


# -------- Parte 2d --------
