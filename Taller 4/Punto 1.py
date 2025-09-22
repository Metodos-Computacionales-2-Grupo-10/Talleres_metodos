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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros
alpha = 0.1
tmax = 150.0
x_min, x_max = -20.0, 20.0
Nx = 800
dx = (x_max - x_min) / (Nx - 1)
x = np.linspace(x_min, x_max, Nx)
dt = 0.05
Nt = int(np.round(tmax / dt)) + 1
t = np.linspace(0, tmax, Nt)

# Potencial: oscilador armónico
V = -x**2 / 50.0

# Condición inicial: paquete gaussiano con momento inicial
x0 = 10.0
k0 = 2.0
psi0 = np.exp(-2.0 * (x - x0)**2) * np.exp(-1j * k0 * x)
norm0 = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
psi0 = psi0 / norm0

# Matriz de derivadas (Neumann)
main = -2.0 * np.ones(Nx)
off = 1.0 * np.ones(Nx - 1)
D = sp.diags([off, main, off], offsets=[-1, 0, 1], format='csc') / dx**2
D = D.tolil()
D[0,0], D[0,1] = -2.0/dx**2, 2.0/dx**2
D[-1,-1], D[-1,-2] = -2.0/dx**2, 2.0/dx**2
D = D.tocsc()

# Hamiltoniano
H = alpha * D - sp.diags(-V, 0, format='csc')
I = sp.identity(Nx, format='csc')
A = I - 1j * dt/2 * H
B = I + 1j * dt/2 * H
A_factor = spla.factorized(A)

# Evolución
psi = psi0.copy().astype(np.complex128)
mu_t, sigma_t = [], []
frames = 400
step_per_frame = Nt // frames

snapshots, times = [], []

for n in range(Nt):
    prob = np.abs(psi)**2
    mu = np.sum(x * prob) * dx
    sigma = np.sqrt(np.sum((x - mu)**2 * prob) * dx)
    mu_t.append(mu)
    sigma_t.append(sigma)

    if n % step_per_frame == 0:
        snapshots.append(prob.copy())
        times.append(n*dt)

    rhs = B.dot(psi)
    psi = A_factor(rhs)
    if (n+1) % 200 == 0:
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# Animación en pantalla
fig, ax = plt.subplots()
line, = ax.plot(x, snapshots[0])
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(map(np.max, snapshots))*1.1)
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x,t)|²')

def animate(i):
    line.set_ydata(snapshots[i])
    ax.set_title(f't = {times[i]:.1f}')
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), blit=True)

# Mostrar animación y luego la gráfica de μ ± σ
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, mu_t, label='μ(t)')
ax2.fill_between(t, np.array(mu_t)-np.array(sigma_t), 
                    np.array(mu_t)+np.array(sigma_t), 
                    alpha=0.3, label='μ ± σ')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Posición media')
ax2.legend()
plt.tight_layout()
plt.show()
