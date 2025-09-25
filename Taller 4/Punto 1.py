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
writer = FFMpegWriter(fps=20, metadata=dict(artist='Thomas Jara'), bitrate=1800)
ani.save("Taller 4/1.a.mp4", writer=writer)

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
plt.savefig("Taller 4/1.a.png")
"Punto 1b: Oscilador cuartico"
# Parámetros básicos
alpha = 0.1                 
x_min, x_max = -20, 20      # Rango espacial
dx = 0.1                    # Paso en el espacio
x = np.arange(x_min, x_max+dx, dx)  # Malla espacial
dt = 0.01                   # Paso de tiempo
T  = 50.0                  # Tiempo total de simulación
nt = int(T/dt)              # Número total de pasos de tiempo


# Potencial
V = (x/5)**4                 


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
writer = FFMpegWriter(fps=20, metadata=dict(artist='Thomas Jara'), bitrate=1800)
ani.save("Taller 4/1.b.mp4", writer=writer)

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
plt.savefig("Taller 4/1.b.png")
''''Punto 1.c: Potencial del sombrero'''
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
writer = FFMpegWriter(fps=20, metadata=dict(artist='Thomas Jara'), bitrate=1800)
ani.save("Taller 4/1.c.mp4", writer=writer)


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
plt.savefig("Taller 4/1.c.png")
# Punto_3.py
# Ecuación de Korteweg–de Vries (KdV):
#   ∂t φ + φ ∂x φ + δ^2 ∂x^3 φ = 0
# Método de líneas: discretizamos en x (derivación espectral) y resolvemos en t con solve_ivp.
# Salidas:
#   - 3_solitones_scipy.png   (comparación t=0, t~T/2, t=T)
#   - 3_solitones_scipy.mp4   (animación de la evolución)

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import solve_ivp


# Parámetros de simulación

L = 40.0           # Longitud del dominio [0, L] con fronteras periódicas
N = 256            # Número de puntos espaciales (potencia de 2 recomendado para FFT)
delta = 0.022      # Parámetro de dispersión (ajústalo para explorar)
Tmax = 40.0        # Tiempo total de simulación
Nt = 600           # Número de muestras de tiempo para guardar/animar (no pasos internos del solver)

# Mallado espacial
x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

# Frecuencias (para derivación espectral)
k = 2.0 * np.pi * fftfreq(N, d=dx)
ik = 1j * k
ik3 = (1j * k) ** 3


# Condición inicial

# Onda coseno: se sabe que "rompe" y se descompone en múltiples solitones.
phi0 = np.cos(2.0 * np.pi * x / L)


# RHS del sistema ODE: dφ/dt = -φ φ_x - δ^2 φ_xxx

def kdv_rhs(t, y):
    
    #t  : tiempo (float) - no se usa directamente, pero es requerido por solve_ivp
    #y  : φ(x,t) discretizada (shape: N,)
    
    phi = y
    phi_hat = fft(phi)

    # Derivadas espaciales vía FFT (periodicidad)
    phi_x   = np.real(ifft(ik * phi_hat))
    phi_xxx = np.real(ifft(ik3 * phi_hat))

    # KdV:
    # ∂t φ = - φ ∂x φ - δ^2 ∂x^3 φ
    return -(phi * phi_x) - (delta**2) * phi_xxx

# Malla temporal para muestrear la solución (el solver usará pasos internos adaptativos)
t_eval = np.linspace(0.0, Tmax, Nt)


# Integración temporal con solve_ivp (RK45 adaptativo)

sol = solve_ivp(fun=kdv_rhs,t_span=(0.0, Tmax),y0=phi0,t_eval=t_eval,method="RK45",rtol=1e-6,atol=1e-9,vectorized=False)



# sol.y tiene shape (N, Nt): columnas = φ(x, t_i)
phi_evol = sol.y
times = sol.t


# Gráfica comparativa: t=0, t~T/2, t=T

idx0 = 0
idxm = len(times)//2
idxf = -1

plt.figure(figsize=(8, 4))
plt.plot(x, phi_evol[:, idx0], label=f"t={times[idx0]:.2f}")
plt.plot(x, phi_evol[:, idxm], label=f"t={times[idxm]:.2f}")
plt.plot(x, phi_evol[:, idxf], label=f"t={times[idxf]:.2f}")
plt.xlabel("x")
plt.ylabel("ϕ(x,t)")
plt.title("Evolución KdV (solve_ivp + diferenciación espectral)")
plt.legend()
plt.tight_layout()
plt.savefig("Punto3", dpi=200)
plt.close()


# Animación con FuncAnimation + FFMpegWriter

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, phi_evol[:, 0])
ax.set_xlim(x[0], x[-1])
# Límites verticales seguros; puedes ajustarlos según tus parámetros
ymin = min(phi_evol.min(), -2.0)
ymax = max(phi_evol.max(),  2.0)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("x")
ax.set_ylabel("ϕ(x,t)")
ttl = ax.set_title(f"Solitones KdV — t = {times[0]:.2f}")

def animate(i):
    line.set_ydata(phi_evol[:, i])
    ttl.set_text(f"Solitones KdV — t = {times[i]:.2f}")
    return line, ttl
ani = FuncAnimation(fig, animate, frames=len(times), interval=30, blit=False)
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save("3_solitones_scipy.mp4", writer=writer, dpi=200)
