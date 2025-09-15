import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import curve_fit
from numba import njit
"""2. Balistica"""
# Constantes
m = 10.01  # kg
A = 1.642
B = 40.624
C = 2.36
g = 9.7732  # m/s^2
v_0 = np.linspace(0, 140, 100)  
angulos = np.linspace(10, 80, 100)  

#Coeficiente de fricción
def beta(y):
    return A * (1 - y / B) ** C

# Modelo de ecuaciones
###Resolviendo sumatoria de fuerzas=ma y diviendo por m
def modelo(t, estado):
    x, y, vx, vy = estado
    v = np.sqrt(vx**2 + vy**2)
    ax = -beta(y) * v * vx / m
    ay = -g - beta(y) * v * vy / m
    return [vx, vy, ax, ay]

# Evento donde acaba la simulación (cuando y=0), aca me ayudo chat y la guia de santiago con .terminal y .direction
def hit_ground(t, estado):
    return estado[1]
hit_ground.terminal = True
hit_ground.direction = -1

alcances = []
thetas_xmax = []

for vel in v_0:
    x_max = 0
    theta_max = 0
    for theta_deg in angulos:
        theta_rad = np.deg2rad(theta_deg)
        estado_inicial = [0, 0,
                          vel * np.cos(theta_rad),
                          vel * np.sin(theta_rad)]

        sol = solve_ivp(modelo, t_span=(0, 200),
                        y0=estado_inicial,
                        events=hit_ground, max_step=0.1)

        if sol.status == 1 and sol.t_events[0].size > 0:
            x_land = sol.y[0, -1]
            if x_land > x_max:
                x_max = x_land
                theta_max = theta_deg  # guardamos en GRADOS
    alcances.append(x_max)
    thetas_xmax.append(theta_max)
    plt.scatter(vel, x_max, color='blue')
    #plt.scatter(vel, theta_max, color='red')
plt.xlabel('Velocidad inicial (m/s)')
plt.ylabel('Alcance máximo (m)')
plt.title('Alcance máximo vs Velocidad inicial')
plt.savefig('Taller 3/2.a.pdf')
plt.close()
"""2.b Atinar a un objetivo
Escriba una función  angle_to_hit_target(v0,target_x,target_y)  que dada una veloci0
dad inicial fija, encuentre el ángulo necesario 𝜃0 para que el cañón atine a un objetivo en las
coordenadas  target_x,target_y , ambas positivas.
Debería probar con varios objetivos, graficar la trayectoria y asegurarse de que su código
funciona, pues no queremos decepcionar al general. No necesita guardar estas gráficas"""

def angle_to_hit_target(v0, target_x, target_y):
    if target_x>0 and target_y>=0:
        angulo_resultado = []
        sol_hit = None
        angulos = np.linspace(10, 80, 100)  # en grados
        tolerancia = 0.05 # margen de error
        for angulo in angulos:
            angulo_rad = np.deg2rad(angulo)
            estado_inicial = [0, 0,
                              v0 * np.cos(angulo_rad),
                              v0 * np.sin(angulo_rad)]

            sol = solve_ivp(modelo, t_span=(0, 200),
                            y0=estado_inicial,
                            events=hit_ground, max_step=0.1)
            
            trayectoria_x = sol.y[0]
            trayectoria_y = sol.y[1]
            ###No pude resolver para un punto especifico, pues se necesita muchos puntos
            ###entonces lo que hice fue buscar si en la trayectoria hay un punto entre pos_objetivo-tolerancia y pos_objetivo+tolerancia
            if np.any((trayectoria_x <= target_x + tolerancia) & (trayectoria_x >= target_x - tolerancia)) and np.any((trayectoria_y <= target_y + tolerancia) & (trayectoria_y >= target_y - tolerancia)):
                angulo_resultado.append(angulo)
                sol_hit = sol
    else:
        #print("Objetivo fuera del espacio")
        return None, None, None

    if sol_hit is None:
        #print("No se encontró ningún ángulo que alcance el objetivo con esa velocidad.")
        return None, None, None
    
    return angulo_resultado, sol_hit.y[0], sol_hit.y[1]

    
##PRUEBA y VISUALIZACION
#angulo_hit,x,y = angle_to_hit_target(40,12,0)
#plt.plot(x,y,label='trayectoria')
#plt.scatter(12,0,label='Target',color= 'r')
#plt.show()

"""2.c Varias opciones para disparar
Dado un objetivo (𝑥, 𝑦) existen varias combinaciones de 𝑣0 y 𝜃0 que hacen que el disparo
llegue al objetivo. Caracterice este conjunto de soluciones. ¿Qué dimensión tiene? ¿Se le ocurre
alguna parametrización?
Cree una gráfica ( 2.c.pdf ) de 𝜃0 vs 𝑣0 que muestre dónde están las condiciones iniciales que
atinan al objetivo en (𝑥, 𝑦) = (12m, 0)
"""

for vel in v_0:
    angulos_hit, x, y = angle_to_hit_target(vel, 12, 0)
    if angulos_hit is not None and len(angulos_hit) > 0:
        for angulo in angulos_hit:
            plt.scatter(vel, angulo, color='blue')
plt.xlabel('Velocidad inicial (m/s)')
plt.ylabel('Ángulo de disparo (grados)')
plt.title('Ángulo de disparo vs Velocidad inicial para alcanzar (12m, 0)')
plt.savefig('Taller 3/2.c.pdf')
        
