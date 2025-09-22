"""import numpy as np
import matplotlib.pyplot as plt
from numba import njit
#PARAMS
alpha=0.1
t=np.linspace(0,150,150)
x=np.linspace(-20,20,100)
#CONDICIONES NEUMANN 
def condiciones_frontera(U):
    U[0] = 0 
    U[-1] = 0 
    return U

V=lambda x: -x**2/50
psi_0=np.exp(-2(x-10)**2) *np.exp(-1j*2*x)
U=np.zeros((len(t),len(x))) *np.nan
U=condiciones_frontera(U)

@njit("void(f8[:,:], f8[:,:], f8[:], f8, f8)")
def evolve_schrodinger(U, V, pot, dx, dt):
    Nx = U.shape[1]
    Nt = U.shape[0]
    for n in range(Nt-1):       # bucle temporal
        for j in range(1, Nx-1): # bucle espacial interior
            # Segunda derivada (laplaciano 1D)
            d2U = (U[n,j-1] - 2*U[n,j] + U[n,j+1]) / dx**2
            d2V = (V[n,j-1] - 2*V[n,j] + V[n,j+1]) / dx**2
            # Ecuación acoplada
            U[n+1,j] = U[n,j] + dt*( -d2V + pot[j]*V[n,j] )
            V[n+1,j] = V[n,j] + dt*(  d2U - pot[j]*U[n,j] )
        # Condiciones de frontera de Neumann: ∂ψ/∂x = 0
        U[n+1,0] = U[n+1,1]
        U[n+1,-1] = U[n+1,-2]
        V[n+1,0] = V[n+1,1]
        V[n+1,-1] = V[n+1,-2]"""
        
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

# Parámetros
alpha = 0.1
x_min, x_max = -20, 20
dx = 0.1
x = np.arange(x_min, x_max+dx, dx)
N = len(x)

dt = 0.01     # paso de tiempo
T  = 150.0    # tiempo total
nt = int(T/dt)

# Potencial oscilador armónico
V = x**2/50

# Condición inicial: paquete gaussiano centrado en x0=10 con momento k0=-2
x0 = 10.0
k0 = -2.0
psi0 = np.exp(-2*(x-x0)**2) * np.exp(1j*k0*x)
psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, x))  # normalizar


@njit
def laplacian_neumann(psi, dx):
    N = len(psi)
    d2 = np.zeros(N, dtype=np.complex128)
    for i in range(1, N-1):
        d2[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
    d2[0]  = (psi[1] - psi[0]) * 2 / dx**2
    d2[-1] = (psi[-2] - psi[-1]) * 2 / dx**2
    return d2


@njit
def schrodinger_rhs(psi, alpha, V, dx):
    return 1j*(alpha*laplacian_neumann(psi, dx) - V*psi)


@njit
def rk4_step(psi, dt, alpha, V, dx):
    k1 = schrodinger_rhs(psi, alpha, V, dx)
    k2 = schrodinger_rhs(psi + 0.5*dt*k1, alpha, V, dx)
    k3 = schrodinger_rhs(psi + 0.5*dt*k2, alpha, V, dx)
    k4 = schrodinger_rhs(psi + dt*k3, alpha, V, dx)
    return psi + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# Evolución temporal (guardando cada "skip" pasos)
skip = 10  # guarda cada 50 pasos -> avanza 0.5 unidades de tiempo por frame
psi = psi0.copy()
frames = []
times  = []

for n in range(nt):
    if n % skip == 0:
        frames.append(np.abs(psi)**2)
        times.append(n*dt)
    psi = rk4_step(psi, dt, alpha, V, dx)

frames = np.array(frames)
times  = np.array(times)

# Animación
fig, ax = plt.subplots(figsize=(8,4.5))
line, = ax.plot(x, frames[0])
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, np.max(frames)*1.1)
ax.set_xlabel("x")
ax.set_ylabel("|ψ(x,t)|²")
title = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def update(i):
    line.set_ydata(frames[i])
    title.set_text(f"t = {times[i]:.1f}")
    return line, title

ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True, interval=30)
plt.show()
