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

# -----------------------------
# INTEGRAR para una energía de prueba E
# -----------------------------
def integrar(E):
    # 1) calculamos los puntos de giro x1, x2 (donde V(x)=E)
    puntos = root(E)
    if puntos is None:
        # Si no hay puntos de giro, no hay un pozo clásico acotado para esa E
        return None, None, None

    x1, x2 = puntos
    # 2) Según la guía: integrar desde x1 - 2 hasta x2 + 1
    x_span = (x1 - 2, x2 + 1)  
    x_eval = np.linspace(*x_span, 800)  # puntos donde queremos la solución

    # 3) condiciones iniciales:
    #    psi(x_start) = 0   
    #    psi'(x_start) = 1e-5 
    sol = solve_ivp(
        schrodinger, x_span, [0, 1e-5], args=(E,),
        t_eval=x_eval, max_step=0.01  # max_step = 0.01 como piden
    )

    # sol.y[0] es psi(x) en los puntos sol.t
    psi = sol.y[0]
    # devolvemos los puntos x (sol.t), la psi completa y el valor final psi[-1]
    return sol.t, psi, psi[-1]

# -----------------------------
# BÚSQUEDA DE ENERGÍAS (método shooting)
# -----------------------------
# - "E_vals" son las energías de prueba (cada E es un 'epsilon' de prueba).
# - Para cada E integramos y guardamos psi en la frontera derecha (psi_end).
# - Si psi_end cambia de signo entre dos E consecutivas, hay una raíz en ese
#   intervalo => hay una energía ligada. Entonces usamos brentq para refinarla.
E_vals = np.linspace(-0.999, -0.01, 300)  # rango de epsilones de prueba
psi_endpoints = []

for E in E_vals:
    x, psi, psi_end = integrar(E)
    if x is not None:
        psi_endpoints.append(psi_end)
    else:
        # Si integrar devolvió None significa que no había pozo clásico; lo marcamos como NaN
        psi_endpoints.append(np.nan)

psi_endpoints = np.array(psi_endpoints)

# -----------------------------
# Detectar CRUCES DE SIGNO en psi_final(E)
# -----------------------------
# np.diff(np.sign(...)) detecta cambios en el signo de la lista de valores.
# Usamos np.nan_to_num para convertir NaN a 0 antes de tomar el signo.
# Nota: convertir NaN a 0 puede introducir ceros que NO son raíces físicas,
# pero luego la refinación con brentq filtrará la mayoría de falsos positivos.
idx_crossings = np.where(np.diff(np.sign(np.nan_to_num(psi_endpoints))))[0]

# -----------------------------
# REFINAR CON brentq
# -----------------------------
# Ahora para cada cruce usamos brentq sobre la función f(E) = psi_final(E)
# brentq busca un E en el intervalo [E_vals[i], E_vals[i+1]] tal que f(E)=0.
#
# Comentario sobre brentq (en palabras sencillas):
#   - brentq es un método numérico que encuentra una raíz de una función
#     cuando se le da un intervalo donde la función cambia de signo.
#   - Aquí le damos dos energías que "encierran" una raíz y le pedimos que
#     encuentre la energía exacta donde psi_final(E)=0.
bound_state_energies = []
for i in idx_crossings:
    try:
        # lambda energy: integrar(energy)[2] -> devuelve psi_final para la energía
        energy_root = brentq(lambda energy: integrar(energy)[2], E_vals[i], E_vals[i+1])
        bound_state_energies.append(energy_root)
    except ValueError:
        # Si brentq falla (por ejemplo porque la integral devolvió None en el intervalo),
        # capturamos el error y seguimos con el siguiente cruce.
        pass

# -----------------------------
# GRAFICAR POTENCIAL y FUNCIONES DE ONDA (normalizadas)
# -----------------------------
plt.figure(figsize=(9,6))
x_plot = np.linspace(5, 15, 500)
plt.plot(x_plot, V(x_plot), 'k', label="V(x)")

escala = 0.05  # cuánto "estiramos" psi para que se vea bien cuando la dibujamos encima de E
for n, E in enumerate(bound_state_energies):
    x, psi, _ = integrar(E)
    if x is not None:
        # Normalización: ∫ psi^2 dx = 1 (aproximada con trapz)
        norm = np.sqrt(np.trapz(psi**2, x))
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
plt.show()



# ------------------------------
# Comparación con valores teóricos
# ------------------------------

# Parámetro λ
lam = 1 / (a * hbar)

# Energías teóricas (hasta que dejen de ser positivas dentro del pozo)
E_teoricas = []
n_vals = range(len(bound_state_energies))
for n in n_vals:
    E_n = (2*lam - (n + 0.5)) * (n + 0.5) / (lam**2)
    if E_n < 0:  # solo los estados ligados
        E_teoricas.append(E_n)
    else:
        break

# Ajustamos la lista a las energías numéricas obtenidas
n_vals = range(len(E_teoricas))
import numpy as np

# Construir tabla en formato numérico
tabla = []
for n, (E_num, E_teo) in enumerate(zip(bound_state_energies, E_teoricas)):
    diff_pct = abs((E_num - E_teo) / E_teo) * 100
    tabla.append([n, E_num, E_teo, diff_pct])

tabla = np.array(tabla)

# Guardar como archivo de texto
np.savetxt(
    "Taller 3/1.c.txt",
    tabla,
    header="n   E_numérico   E_teórico   Diferencia(%)",
    fmt=["%d", "%.6f", "%.6f", "%.3f"],
    delimiter="\t"
)
