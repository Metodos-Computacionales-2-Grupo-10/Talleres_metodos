#1
import numpy as np
import matplotlib.pyplot as plt

def ising_step(grid, J=1.0, h=0.0, Beta=0.5):
    Lx, Ly = grid.shape
    i = np.random.randint(Lx)
    j = np.random.randint(Ly)
    s = grid[i, j]

    
    up    = grid[(i-1)%Lx, j]
    down  = grid[(i+1)%Lx, j]
    left  = grid[i, (j-1)%Ly]
    right = grid[i, (j+1)%Ly]

    dE = 2 * s * (J*(up+down+left+right) + h)
    dm = -2 * s
    
    if dE <= 0 or np.random.rand() < np.exp(-Beta*dE):
        grid[i, j] *= -1
        return grid, dE, dm   
    else:
        return grid, 0, 0
def simulate_ising(L=20, Beta=0.5, steps=10000, J=1.0, h=0.0):
    grid = np.random.choice([-1, 1], size=(L, L)) 
    energies = []
    E0 = 0
    magnes = []
    for i in range(L):
        for j in range(L):
            s = grid[i,j]
            up    = grid[(i-1)%L, j]
            down  = grid[(i+1)%L, j]
            left  = grid[i, (j-1)%L]
            right = grid[i, (j+1)%L]
            E0 += -J * s * (up+down+left+right)
    E0 = E0/2   
    m0 = np.sum(grid)
    cumulative_E = E0
    cumulative_m = m0
    energies.append(E0 / (4*L*L))
    magnes.append(m0 / (L*L))
    for t in range(steps):
        grid, dE, dm = ising_step(grid, J=J, h=h, Beta=Beta)
        cumulative_E += dE
        cumulative_m += dm
        energies.append(cumulative_E / (4*L*L))
        magnes.append(cumulative_m / (L*L))

    return grid, energies, magnes

grid, energies, magnes = simulate_ising(L=50, Beta=0.5, steps=200000)

plt.figure(figsize=(8,5))
plt.plot(energies, label="Energía promedio (normalizada)", color="tab:blue")
plt.plot(magnes, label="Magnetización (normalizada)", color="tab:red")

plt.xlabel("Pasos Monte Carlo")
plt.ylabel("Magnitud normalizada")
plt.title("Ising 2D, J=1, β=1/2")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("Taller 5/1a.pdf")

#1.b
def recorrer_betas(betas_max=1.0, L=50, _equil=200,_meas=500, n_points=10, J=1.0, h=0.0):
    betas_array = np.linspace(0.0, betas_max, n_points)
    Cvs = []
    grid = np.random.choice([-1,1], size=(L,L))

    for beta in betas_array:
        for _ in range(_equil * L * L):
            grid, _, _ = ising_step(grid, J=J, h=h, Beta=beta)
        energies = []
        cumulative_E = 0
        for i in range(L):
            for j in range(L):
                s = grid[i,j]
                up    = grid[(i-1)%L, j]
                down  = grid[(i+1)%L, j]
                left  = grid[i, (j-1)%L]
                right = grid[i, (j+1)%L]
                cumulative_E += -J * s * (up+down+left+right)
        cumulative_E /= 2   

        for _ in range(_meas * L * L):
            grid, dE, _ = ising_step(grid, J=J, h=h, Beta=beta)
            cumulative_E += dE
            energies.append(cumulative_E / (4*L*L))

        
        E_mean = np.mean(energies)
        E2_mean = np.mean(np.array(energies)**2)
        Cv = (beta**2) * (L**2) * (E2_mean - E_mean**2)
        Cvs.append(Cv)

    return betas_array, np.array(Cvs)

betas_array, Cvs = recorrer_betas(betas_max=1.0, L=50, _equil=250, _meas=1000, n_points=20)


plt.plot(betas_array, Cvs, '-o')
plt.axvline(0.5*np.log(1+np.sqrt(2)), color='k', linestyle='--', label=r'$\beta_c$ de Onsager')
plt.xlabel("Beta")
plt.ylabel("Cv")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("Taller 5/1b.pdf")
