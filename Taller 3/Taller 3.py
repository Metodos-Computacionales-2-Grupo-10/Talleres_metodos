import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import curve_fit
from numba import njit


# 1.a — Lotka–Volterra


def lv_rhs(t, z, alpha, beta, gamma, delta):
    x, y = z #vector estado 
    dx = alpha*x - beta*x*y
    dy = -gamma*y + delta*x*y #basicamente las ecuaciones y devuelve x y y punto
    return [dx, dy]

alpha, beta, gamma, delta = 2.0, 1.5, 0.3, 0.4
x0, y0 = 3.0, 2.0
t0, tf = 0.0, 50.0 #basicamente todas las ctes y que se integra en un itervalo de tiempo entre 0 y 50 
t_eval = np.linspace(t0, tf, 10001) #evualuar 10001 puntos uniformes para graficar suave 

sol = solve_ivp(fun=lambda t, z: lv_rhs(t, z, alpha, beta, gamma, delta),t_span=(t0, tf), y0=[x0, y0],t_eval=t_eval, method="RK45")
t = sol.t
x, y = sol.y #solución numérica 
V = delta*x - gamma*np.log(np.maximum(x, 1e-12)) + beta*y - alpha*np.log(np.maximum(y, 1e-12)) #cantidad conservada, 1e-12 evita ln(0)

fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
axes[0].plot(t, x); axes[0].set_title("1.a Lotka/Volterra: x(t)")
axes[1].plot(t, y); axes[1].set_title("1.a Lotka/Volterra: y(t)")
axes[2].plot(t, V); axes[2].set_title("1.a Cantidad conservada V(t)")
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1a.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)


# 1.b — Landau 2D con E(x), Bz


def landau_rhs(t, y, q, m, E0, B0, k):
    x, y_, vx, vy = y
    Fx = q*E0*(np.sin(k*x) + k*x*np.cos(k*x))
    dvx = (Fx + q*B0*vy)/m
    dvy = (-q*B0*vx)/m
    return [vx, vy, dvx, dvy]

q, m = 7.5284, 3.8428
B0, E0, k = 0.438, 0.7423, 1.0014
x0, y0, vx0, vy0 = 0.1, 0.0, 0.0, 0.2
t0, tf = 0.0, 30.0
t_eval = np.linspace(t0, tf, 60001) #todo exactamente igual al primero 

sol = solve_ivp(fun=lambda t, Y: landau_rhs(t, Y, q, m, E0, B0, k),t_span=(t0, tf), y0=[x0, y0, vx0, vy0],t_eval=t_eval, method="RK45")
t = sol.t
x, y, vx, vy = sol.y
K = 0.5*m*(vx**2 + vy**2)
U = - q*E0*x*np.sin(k*x)
E = K + U
Pi_y = m*vy - q*B0*x #cantidades fisicas a monitorear, este es el momento conjugado en y con B uniforme 

fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
axes[0].plot(t, x);    axes[0].set_title("1.b Landau: x(t)")
axes[1].plot(t, y);    axes[1].set_title("1.b Landau: y(t)")
axes[2].plot(t, Pi_y); axes[2].set_title("1.b Momento conjugado Π_y(t)")
axes[3].plot(t, E);    axes[3].set_title("1.b Energía total E(t)")
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1b.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)


# 1.c — Dos cuerpos gravitacionales


def two_body_rhs(t, y, m=1.7, G=1.0):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    dr = r1 - r2
    dist = np.sqrt(dr[0]**2 + dr[1]**2)
    dist3 = dist**3 if dist > 1e-12 else 1e-12 #esto me lo dio chat para evitar la divicion por 0 en la ley de gravitación 
    a1 = -G*m*dr/dist3
    a2 = -G*m*(-dr)/dist3
    return [vx1, vy1, a1[0], a1[1], vx2, vy2, a2[0], a2[1]]

m, G = 1.7, 1.0
r1_0, v1_0 = [0.0, 0.0], [0.0, 0.5]
r2_0, v2_0 = [1.0, 1.0], [0.0,-0.5]
t0, tf = 0.0, 10.0
t_eval = np.linspace(t0, tf, 20001)

y0 = [r1_0[0], r1_0[1], v1_0[0], v1_0[1], r2_0[0], r2_0[1], v2_0[0], v2_0[1]]

sol = solve_ivp(fun=lambda t, y: two_body_rhs(t, y, m=m, G=G),t_span=(t0, tf), y0=y0,t_eval=t_eval, method="RK45")
t = sol.t
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = sol.y
R1 = np.vstack((x1, y1)).T
R2 = np.vstack((x2, y2)).T
V1 = np.vstack((vx1, vy1)).T
V2 = np.vstack((vx2, vy2)).T #convierte un (vector) en un array de puntos vertical
dr = R1 - R2
dist = np.linalg.norm(dr, axis=1)
U = - G*m*m / np.maximum(dist, 1e-12)
K = 0.5*m*(np.sum(V1**2, axis=1) + np.sum(V2**2, axis=1))
E = K + U
Lz = m*(R1[:,0]*V1[:,1] - R1[:,1]*V1[:,0]) + m*(R2[:,0]*V2[:,1] - R2[:,1]*V2[:,0])

fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
axes[0].plot(t, x1, label="x1"); axes[0].plot(t, x2, label="x2", alpha=0.85)
axes[1].plot(t, y1, label="y1"); axes[1].plot(t, y2, label="y2", alpha=0.85)
axes[2].plot(t, E);  axes[2].set_title("1.c Energía total E(t)")
axes[3].plot(t, Lz); axes[3].set_title("1.c Momento angular total Lz(t)")
axes[0].set_title("1.c Binario: x1(t), x2(t)"); axes[0].legend()
axes[1].set_title("1.c Binario: y1(t), y2(t)"); axes[1].legend()
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1c.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

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
        tolerancia = 0.1 # margen de error
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
            #####Correccion (tomaba trayectorias que no llegaban al target) chat me recomendo hacerlo asi
            distancias = np.sqrt((trayectoria_x - target_x)**2 + (trayectoria_y - target_y)**2)

            if np.any(distancias < tolerancia):
                angulo_resultado.append(angulo)
                sol_hit = sol
                #plt.plot(trayectoria_x, trayectoria_y, label=f'Trayectoria ángulo {angulo:.2f}°')
    else:
        return None, None, None

    if sol_hit is None:
        return None, None, None
    #plt.scatter(target_x, target_y, label='Target', color='r')
    #plt.legend()
    #plt.show()
    return angulo_resultado, sol_hit.y[0], sol_hit.y[1]
"""ESTA FUNCION FUNCIONA PERO TIENE ERRORES NO SE PORQUE"""
    
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
        
"""3. Molecula diatomica"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


#Parametros
hbar = 0.1
a = 0.8
x0 = 10.0

# Potencial de Morse
def V(x):
    return (1 - np.exp(a*(x - x0)))**2 - 1

# Ecuación de Schrödinger
def schrodinger(x, y, E):
    # y[0] = psi(x)
    # y[1] = psi'(x)
    return [y[1], (1/hbar**2) * (V(x) - E) * y[0]]


# Esta función busca raíces de V(x)-E en un rango dado
# Cada vez que detecta un cambio de signo entre puntos vecinos, usa brentq para encontrar la raíz con precisión.
def root(E, x_range=(0, 20), npts=1000):
    xs = np.linspace(*x_range, npts)
    vals = V(xs) - E
    roots = []
    for i in range(len(xs)-1):
        # Si el producto es negativo hay cambio de signo => raíz entre xs[i] y xs[i+1]
        if vals[i]*vals[i+1] < 0:
            try:
                # brentq encuentra la raíz de la función en el intervalo dado
                r = brentq(lambda xx: V(xx)-E, xs[i], xs[i+1])
                roots.append(r)
            except ValueError:
                # Si brentq falla en este intervalo lo ignoramos y seguimos
                pass
    # Si encontramos menos de 2 raíces no hay una región acotada clásica (no se puede aplicar la guía)
    if len(roots) < 2:
        return None
    # Devolvemos la primera y la última raíz encontradas (x1 y x2)
    return roots[0], roots[-1]


# INTEGRAR para una energía de prueba E

def integrar(E):
    # 1) calculamos los puntos x1, x2 donde V(x)=E
    puntos = root(E)
    if puntos is None:
        # Si no hay puntos, retorna None
        return None, None, None

    x1, x2 = puntos
    # 2) Según la guía: integrar desde x1 - 2 hasta x2 + 1
    x_interval = (x1 - 2, x2 + 1)  
    x_eval = np.linspace(*x_interval, 800)  # puntos donde queremos la solución

    # 3) condiciones iniciales:
    #    psi(x_start) = 0   
    #    psi'(x_start) = 1e-5 
    sol = solve_ivp(
        schrodinger, x_interval, [0, 1e-5], args=(E,),
        t_eval=x_eval, max_step=0.01  # max_step = 0.01 como piden
    )

    # sol.y[0] es psi(x) en los puntos sol.t
    psi = sol.y[0]
    # devolvemos los puntos x (sol.t), la psi completa y el valor final psi[-1]
    return sol.t, psi, psi[-1]

# -----------------------------
# BÚSQUEDA DE ENERGÍAS (shooting simplificado)
# -----------------------------

# Paso 1: generar energías de prueba (epsilon de prueba)
# Aquí probamos valores de energía entre -0.999 y -0.01
E_pruebas = np.linspace(-0.999, -0.01, 100)

# Aquí vamos a guardar los valores de psi en el borde derecho
psi_finales = []

# Paso 2: integrar para cada energía
for E in E_pruebas:
    x, psi, psi_end = integrar(E)   # resolvemos la ecuación
    if x is not None:
        # Guardamos el valor final de la función de onda
        psi_finales.append(psi_end)
    else:
        # Si no hay solución válida guardamos 0
        psi_finales.append(0)


psi_finales = np.array(psi_finales)

# Paso 3: buscar intervalos donde la función cambió de signo
# Eso significa que psi_final pasó de + a - o de - a +, 
# lo cual indica que en medio hay una raíz.
idx_cruces = []
for i in range(len(E_pruebas)-1):
    if psi_finales[i] * psi_finales[i+1] < 0:
        idx_cruces.append(i)

# Paso 4: refinar con brentq
# brentq encuentra con precisión el valor de energía donde psi_final(E) = 0
energias_ligadas = []
for i in idx_cruces:
    try:
        E_raiz = brentq(lambda E: integrar(E)[2], E_pruebas[i], E_pruebas[i+1])
        energias_ligadas.append(E_raiz)
    except:
        # Si por algún motivo falla, pasamos al siguiente
        pass


# -----------------------------
# GRAFICAR POTENCIAL y FUNCIONES DE ONDA (normalizadas)
# -----------------------------
plt.figure(figsize=(9,6))
x_plot = np.linspace(5, 15, 500)
plt.plot(x_plot, V(x_plot), 'k', label="V(x)")

escala = 0.05  
for n, E in enumerate(energias_ligadas):
    x, psi, _ = integrar(E)
    if x is not None:
        # Normalización: ∫ psi^2 dx = 1 
        norm = np.sqrt(np.trapezoid(psi**2, x))
        psi /= norm
        # Dibujamos una línea punteada donde está la energía E
        plt.hlines(E, 0, 15, linestyles="dashed", colors="gray", alpha=0.7)
        # Dibujamos psi normalizada, desplazada verticalmente a la energía E
        plt.plot(x, E + escala*psi)

plt.xlabel("x")
plt.ylabel("Energía")
plt.title("Potencial de Morse y estados ligados (Shooting)")
plt.ylim(-1.1, 0.1)
plt.xlim(0, 15)
plt.legend()
plt.grid()
plt.savefig("Taller 3/3.pdf")
plt.close()



# Comparación con valores teóricos

import numpy as np

# Parámetros
hbar = 0.1
a = 0.8
x0 = 10.0

# Parámetro λ
lam = 1 / (a * hbar)


# Energías teóricas (hasta que dejen de ser negativas dentro del pozo)
E_teoricas = []
n_vals = range(len(energias_ligadas))
for n in n_vals:
    E_n = -1 + (n + 0.5) * hbar * a * np.sqrt(2) - (hbar**2 * a**2 / 8) * (n + 0.5)**2
    if E_n < 0:  # solo los estados ligados (energías negativas)
        E_teoricas.append(E_n)
    else:
        break

# Ajustamos la lista a las energías numéricas obtenidas
E_teoricas = E_teoricas[:len(energias_ligadas)]

# Construir tabla en formato numérico
tabla = []
#print("Comparación de energías:")
#print("n\tE_numérico\tE_teórico\tDiferencia(%)")
#print("-" * 50)

for n, (E_num, E_teo) in enumerate(zip(energias_ligadas, E_teoricas)):
    diff_pct = abs((E_num - E_teo) / abs(E_teo)) * 100  # Usamos valor absoluto para el porcentaje
    #print(f"{n}\t{E_num:.6f}\t{E_teo:.6f}\t{diff_pct:.3f}%")
    tabla.append([n, E_num, E_teo, diff_pct])

tabla = np.array(tabla)

# Guardar como archivo de texto
np.savetxt(
    "Taller 3/3.txt",  
    tabla,
    header="n\tE_numérico\tE_teórico\tDiferencia(%)",
    fmt=["%d", "%.6f", "%.6f", "%.3f"],
    delimiter="\t"
)
