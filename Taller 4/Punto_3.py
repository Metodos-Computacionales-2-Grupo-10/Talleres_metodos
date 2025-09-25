
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
Nt = 600           # Número de muestras de tiempo para (animar no pasos internos del solver)

# Mallado espacial
x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

# Frecuencias (para derivación espectral)
k = 2.0 * np.pi * fftfreq(N, d=dx)
ik = 1j * k
ik3 = (1j * k) ** 3

# Condición inicial

# Onda coseno la cual se sabe que "rompe" y se descompone en múltiples solitones.
phi0 = np.cos(2.0 * np.pi * x / L)

# RHS del sistema ODE: dφ/dt = -φ φ_x - δ^2 φ_xxx
def kdv_rhs(t, y):

    # t  : tiempo (float) - no se usa directamente, pero es requerido por solve_ivp
    # y  : φ(x,t) discretizada (shape: N,)

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
sol = solve_ivp(fun=kdv_rhs, t_span=(0.0, Tmax), y0=phi0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9, vectorized=False)

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
plt.savefig("Taller 4/3", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, phi_evol[:, 0])
ax.set_xlim(x[0], x[-1])
# Límites verticales
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
ani.save("Taller 4/3_solitones_scipy.mp4", writer=writer, dpi=200)
