import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parámetros
L = 80.0
N = 600
dx = L / N
x = np.linspace(0, L, N, endpoint=False)

dt = 0.001
n_steps = 3000
n_frames = 300   # número de cuadros de la animación

# Condición inicial: solitón aislado
c = 1.0
A = np.sqrt(c) * (x - L/2)
u = 0.82 * (1/np.cosh(A))**2

# Derivadas con diferencias finitas periódicas
def dudx(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

def dudx3(u, dx):
    return (np.roll(u, -2) - 2*np.roll(u, -1) + 2*np.roll(u, 1) - np.roll(u, 2)) / (2 * dx**3)

# RHS KdV
def F(u, dx):
    return -6 * u * dudx(u, dx) - dudx3(u, dx)

# Paso RK4
def rk4_step(u, dt, dx):
    k1 = dt * F(u, dx)
    k2 = dt * F(u + 0.5*k1, dx)
    k3 = dt * F(u + 0.5*k2, dx)
    k4 = dt * F(u + k3, dx)
    return u + (k1 + 2*k2 + 2*k3 + k4)/6

# Evolución temporal
history = []
save_every = n_steps // n_frames

for step in range(n_steps):
    u = rk4_step(u, dt, dx)
    if step % save_every == 0:
        history.append(u.copy())

history = np.array(history)
times = np.linspace(0, dt*n_steps, len(history))

# Gráfica comparativa t=0, mitad, final
plt.figure(figsize=(8,4))
plt.plot(x, history[0], label=f"t={times[0]:.2f}")
plt.plot(x, history[len(history)//2], label=f"t={times[len(times)//2]:.2f}")
plt.plot(x, history[-1], label=f"t={times[-1]:.2f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Evolución KdV (RK4 explícito)")
plt.legend()
plt.tight_layout()
plt.savefig("3_solitones.png", dpi=200)
plt.close()

# Animación
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(x, history[0])
ax.set_xlim(x[0], x[-1])
ax.set_ylim(history.min()*1.2, history.max()*1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ttl = ax.set_title(f"Evolución KdV — t={times[0]:.2f}")

def animate(i):
    line.set_ydata(history[i])
    ttl.set_text(f"Evolución KdV — t={times[i]:.2f}")
    return line, ttl

ani = FuncAnimation(fig, animate, frames=len(times), interval=30, blit=False)
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save("3_solitones.mp4", writer=writer, dpi=200)