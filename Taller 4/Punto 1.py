import numpy as np
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
"""1.a Oscilador Armonico"""
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
        V[n+1,-1] = V[n+1,-2]

