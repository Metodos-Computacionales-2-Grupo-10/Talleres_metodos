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
plt.show()

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

# Evaluación de cada isótopo
isotopos = ["U (Uranio-239)", "Np (Neptunio-239)", "Pu (Plutonio-239)"]
for i, iso in enumerate(isotopos):
    t_est = tiempo_estabilidad(sol.t, sol.y[i], derivs[:, i]) #Si la dereivada es menor a tol, se considera estable
    if t_est is not None:
        print(f"{iso}: Llega a estado estable en t ≈ {t_est:.3f} días, valor ≈ {sol.y[i,-1]:.2f}")
    else:
        print(f"{iso}: NO alcanza estado estable en 30 días, valor final ≈ {sol.y[i,-1]:.2f}")

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
A = 1.0
lambda_U = 0.1
lambda_Np = 0.05
B = 0.1

tiempo_simulacion = 30
num_trayectorias = 5

plt.figure(figsize=(10,6))

for i in range(num_trayectorias):
    tiempos, U, Np, Pu = simulacion_gillespie(tiempo_simulacion, A, lambda_U, lambda_Np, B)
    plt.step(tiempos, Pu, where="post", label=f"Trayectoria {i+1}")

plt.xlabel("Tiempo")
plt.ylabel("Número de Pu")
plt.title("Simulaciones estocásticas con algoritmo de Gillespie")
plt.legend()
plt.show()