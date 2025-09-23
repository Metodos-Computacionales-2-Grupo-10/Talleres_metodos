import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


"Punto 1a: Oscilador armónico"
# Parámetros básicos
alpha = 0.1                 
x_min, x_max = -20, 20      # Rango espacial
dx = 0.1                    # Paso en el espacio
x = np.arange(x_min, x_max+dx, dx)  # Malla espacial
dt = 0.01                   # Paso de tiempo
T  = 100.0                  # Tiempo total de simulación
nt = int(T/dt)              # Número total de pasos de tiempo


# Potencial
V = x**2/50                 


# Condición inicial: paquete gaussiano
x0 = 10                     # Posición inicial del centro del paquete
k0 = -2                     # Momento inicial (hacia la izquierda)
psi0 = np.exp(-2*(x-x0)**2) * np.exp(1j*k0*x)   # Paquete gaussiano con fase
# Normalización para asegurar que ∫ |ψ|² dx = 1
psi0 = psi0 / np.sqrt(np.trapezoid(np.abs(psi0)**2, x))


# Operador Laplaciano con condiciones de Neumann
def laplacian_neumann(psi, dx):
    lap = np.zeros_like(psi, dtype=complex)
    # Fórmula de diferencias finitas en el interior
    lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    # Condiciones de Neumann en los bordes (derivada = 0)
    lap[0] = (psi[1] - psi[0]) / dx**2
    lap[-1] = (psi[-2] - psi[-1]) / dx**2
    return lap

# Definición de la ecuación de Schrödinger
def schrodinger_rhs(psi, alpha, V, dx):
    # dψ/dt = i [ α ∂²ψ/∂x² – V(x)ψ ]
    lap = laplacian_neumann(psi, dx)
    return 1j * (alpha * lap - V * psi)

# Paso de tiempo con Runge-Kutta 4 (RK4)
def rk4_step(psi, dt, alpha, V, dx):
    k1 = schrodinger_rhs(psi, alpha, V, dx)
    k2 = schrodinger_rhs(psi + dt/2*k1, alpha, V, dx)
    k3 = schrodinger_rhs(psi + dt/2*k2, alpha, V, dx)
    k4 = schrodinger_rhs(psi + dt*k3, alpha, V, dx)
    return psi + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ============================
# Inicialización de variables
# ============================
psi = psi0.copy()        # Estado inicial
skip = 20                # Cada cuántos pasos de tiempo guardar datos
frames = []              # Lista para animación
mu_list = []             # Lista de posiciones medias μ(t)
sigma_list = []          # Lista de incertidumbres σ(t)
times = []               # Lista de tiempos guardados

# ============================
# Evolución temporal
# ============================
for n in range(nt):
    # Avanzar un paso de tiempo con RK4
    psi = rk4_step(psi, dt, alpha, V, dx)
    
    # Normalizar para evitar errores numéricos acumulados
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    psi = psi / norm

    # Guardar datos cada "skip" pasos
    if n % skip == 0:
        prob = np.abs(psi)**2                   # Densidad de probabilidad |ψ|²
        mu = np.trapezoid(x * prob, x)              # Posición promedio μ(t) = ⟨x⟩
        sigma2 = np.trapezoid((x - mu)**2 * prob, x)# Varianza σ²(t) = ⟨(x-μ)²⟩
        sigma = np.sqrt(sigma2)                 # Desviación estándar σ(t)
        
        frames.append(prob)                     # Guardar frame para animación
        mu_list.append(mu)                      # Guardar μ(t)
        sigma_list.append(sigma)                # Guardar σ(t)
        times.append(n*dt)                      # Guardar tiempo actual

# ============================
# Animación de la densidad de probabilidad
# ============================
fig, ax = plt.subplots()
line, = ax.plot(x, frames[0], lw=2)
V_scaled = V / np.max(V) * np.max(frames[0])
ax.plot(x, V_scaled, color='grey', label='Potencial')
ax.set_ylim(0, np.max(frames[0]))
ax.set_xlabel("x")
ax.set_ylabel("|ψ|²")
ax.set_title("Evolución temporal del paquete de ondas")

def animate(i):
    line.set_ydata(frames[i])                   # Actualizar densidad de probabilidad
    return line,

ani = FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)
ani.save("1.a.mp4", writer=FFMpegWriter(fps=20))

# ============================
# Gráfico de μ(t) y σ(t)
# ============================
plt.figure()
plt.plot(times, mu_list, label="⟨x⟩ (posición media)", color="blue")
plt.fill_between(times, np.array(mu_list)-np.array(sigma_list),
                 np.array(mu_list)+np.array(sigma_list),
                 color="blue", alpha=0.3, label="⟨x⟩ ± σ")
plt.xlabel("Tiempo")
plt.ylabel("Posición")
plt.legend()
plt.title("Evolución de la posición media y su incertidumbre")
plt.show()
''''Punto 1.b: Potencial del sombrero'''
# Parámetros básicos
alpha = 0.1                 
x_min, x_max = -20, 20      # Rango espacial
dx = 0.1                    # Paso en el espacio
x = np.arange(x_min, x_max+dx, dx)  # Malla espacial
dt = 0.01                   # Paso de tiempo
T  = 50.0                  # Tiempo total de simulación
nt = int(T/dt)              # Número total de pasos de tiempo


# Potencial
V = (1/50)*(((x**4)/100)-x**2)                 


# Condición inicial: paquete gaussiano
x0 = 10                     # Posición inicial del centro del paquete
k0 = -2                     # Momento inicial (hacia la izquierda)
psi0 = np.exp(-2*(x-x0)**2) * np.exp(1j*k0*x)   # Paquete gaussiano con fase
# Normalización para asegurar que ∫ |ψ|² dx = 1
psi0 = psi0 / np.sqrt(np.trapezoid(np.abs(psi0)**2, x))


# Operador Laplaciano con condiciones de Neumann
def laplacian_neumann(psi, dx):
    lap = np.zeros_like(psi, dtype=complex)
    # Fórmula de diferencias finitas en el interior
    lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    # Condiciones de Neumann en los bordes (derivada = 0)
    lap[0] = (psi[1] - psi[0]) / dx**2
    lap[-1] = (psi[-2] - psi[-1]) / dx**2
    return lap

# Definición de la ecuación de Schrödinger
def schrodinger_rhs(psi, alpha, V, dx):
    # dψ/dt = i [ α ∂²ψ/∂x² – V(x)ψ ]
    lap = laplacian_neumann(psi, dx)
    return 1j * (alpha * lap - V * psi)

# Paso de tiempo con Runge-Kutta 4 (RK4)
def rk4_step(psi, dt, alpha, V, dx):
    k1 = schrodinger_rhs(psi, alpha, V, dx)
    k2 = schrodinger_rhs(psi + dt/2*k1, alpha, V, dx)
    k3 = schrodinger_rhs(psi + dt/2*k2, alpha, V, dx)
    k4 = schrodinger_rhs(psi + dt*k3, alpha, V, dx)
    return psi + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ============================
# Inicialización de variables
# ============================
psi = psi0.copy()        # Estado inicial
skip = 20                # Cada cuántos pasos de tiempo guardar datos
frames = []              # Lista para animación
mu_list = []             # Lista de posiciones medias μ(t)
sigma_list = []          # Lista de incertidumbres σ(t)
times = []               # Lista de tiempos guardados

# ============================
# Evolución temporal
# ============================
for n in range(nt):
    # Avanzar un paso de tiempo con RK4
    psi = rk4_step(psi, dt, alpha, V, dx)
    
    # Normalizar para evitar errores numéricos acumulados
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    psi = psi / norm

    # Guardar datos cada "skip" pasos
    if n % skip == 0:
        prob = np.abs(psi)**2                   # Densidad de probabilidad |ψ|²
        mu = np.trapezoid(x * prob, x)              # Posición promedio μ(t) = ⟨x⟩
        sigma2 = np.trapezoid((x - mu)**2 * prob, x)# Varianza σ²(t) = ⟨(x-μ)²⟩
        sigma = np.sqrt(sigma2)                 # Desviación estándar σ(t)
        
        frames.append(prob)                     # Guardar frame para animación
        mu_list.append(mu)                      # Guardar μ(t)
        sigma_list.append(sigma)                # Guardar σ(t)
        times.append(n*dt)                      # Guardar tiempo actual

# ============================
# Animación de la densidad de probabilidad
# ============================
fig, ax = plt.subplots()
line, = ax.plot(x, frames[0], lw=2)
V_scaled = V / np.max(V) * np.max(frames[0])
ax.plot(x, V_scaled, color='grey', linestyle='--', label='Potencial V(x)')
ax.set_ylim(0, np.max(frames[0]))
ax.set_xlabel("x")
ax.set_ylabel("|ψ|²")
ax.set_title("Evolución temporal del paquete de ondas")

def animate(i):
    line.set_ydata(frames[i])                   # Actualizar densidad de probabilidad
    return line,

ani = FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)
# Guardar en mp4 con ffmpeg
ani.save("paquete_ondas.mp4", writer="ffmpeg", fps=20)

# ============================
# Gráfico de μ(t) y σ(t)
# ============================
plt.figure()
plt.plot(times, mu_list, label="⟨x⟩ (posición media)", color="blue")
plt.fill_between(times, np.array(mu_list)-np.array(sigma_list),
                 np.array(mu_list)+np.array(sigma_list),
                 color="blue", alpha=0.3, label="⟨x⟩ ± σ")
plt.xlabel("Tiempo")
plt.ylabel("Posición")
plt.legend()
plt.title("Evolución de la posición media y su incertidumbre")
plt.show()