import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def rk4_step(f, t, y, dt):
    #Runge-Kutta 4 
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt,       y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
def integrate_rk4(f, y0, t0, tf, dt):
    #Integra con RK4 devolviendo t_arr y Y (n_samples x dim).
    n = int(np.ceil((tf - t0)/dt)) + 1
    t = np.linspace(t0, tf, n)
    Y = np.zeros((n, len(y0)), dtype=float)
    Y[0] = y0
    yi = y0.copy()
    ti = t0
    for i in range(1, n):
        yi = rk4_step(f, ti, yi, dt)
        Y[i] = yi
        ti += dt
    return t, Y
def punto_1a():
    # Parámetros
    alpha = 2.0
    beta  = 1.5
    gamma = 0.3
    delta = 0.4
    x0 = 3.0      # decazorros
    y0 = 2.0      # kiloconejos
    t0, tf, dt = 0.0, 50.0, 0.01

    def f_lv(t, z):
        x, y = z
        dx = alpha*x - beta*x*y
        dy = -gamma*y + delta*x*y
        return np.array([dx, dy], dtype=float)

    t, Z = integrate_rk4(f_lv, np.array([x0, y0], dtype=float), t0, tf, dt)
    x, y = Z[:,0], Z[:,1]

    # Cantidad conservada
    # Evitar log(0) con un epsilon pequeño por robustez
    eps = 1e-12
    V = delta*x - gamma*np.log(np.maximum(x, eps)) + beta*y - alpha*np.log(np.maximum(y, eps))

    
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    axes[0].plot(t, x)
    axes[0].set_title("1.a Lotka/Volterra: x(t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x")

    axes[1].plot(t, y)
    axes[1].set_title("1.a Lotka/Volterra: y(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")

    axes[2].plot(t, V)
    axes[2].set_title("1.a Cantidad conservada V(t)")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("V")

    fig.suptitle("Punto 1.a  Sistema depredador/presa (conservación aproximada)")
    fig.savefig("1.a.pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 1.b Problema de Landau (E(x) y Bz constantes)
# Ecuaciones (unidades naturales c=1):
# m * dvx/dt = q*(E0*(sin(kx) + k*x*cos(kx)) + B0*vy)
# m * dvy/dt = q*(-B0*vx)
# dx/dt = vx, dy/dt = vy
# Cantidades: Π_y = m*vy - q*B0*x  ;  Energía: K+U = m/2*(vx^2+vy^2) - q*E0*x*sin(kx)
# ============================================================

def punto_1b():
    # Parámetros dados (unidades naturales)
    c  = 1.0
    q  = 7.5284
    B0 = 0.438
    E0 = 0.7423
    m  = 3.8428
    k  = 1.0014

    # Condiciones iniciales razonables (posición y velocidad)
    # No especificadas en el enunciado: escogemos algo simple.
    x0, y0 = 0.1, 0.0
    vx0, vy0 = 0.0, 0.2

    t0, tf, dt = 0.0, 30.0, 0.001  # paso pequeño para conservar mejor

    def f_landau(t, y):
        x, y_, vx, vy = y
        Ex = E0*np.sin(k*x)
        # Potencial escalar U(x) = - q * E0 * x * sin(k x)
        # dU/dx = -q*E0*(sin(kx) + kx*cos(kx))
        # Por fuerza: m*dvx/dt = -dU/dx + q*(v x B)_x = q*E0*(sin(kx)+k*x*cos(kx)) + q*B0*vy
        dvx = (q/m)*(E0*(np.sin(k*x) + k*x*np.cos(k*x)) + B0*vy)
        dvy = (q/m)*(-B0*vx)
        return np.array([vx, vy, dvx, dvy], dtype=float)

    t, Y = integrate_rk4(f_landau, np.array([x0, y0, vx0, vy0], dtype=float), t0, tf, dt)
    x, y, vx, vy = Y.T

    # Cantidades conservadas:
    Pi_y = m*vy - q*B0*x               # Momento conjugado Py
    K    = 0.5*m*(vx**2 + vy**2)       # Energía cinética
    U    = - q*E0*x*np.sin(k*x)        # Potencial
    E    = K + U                       # Energía total

    # Graficar: solución x(t), y(t), y cantidades Πy(t), E(t)
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
    axes[0].plot(t, x)
    axes[0].set_title("1.b Landau: x(t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x")

    axes[1].plot(t, y)
    axes[1].set_title("1.b Landau: y(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y")

    axes[2].plot(t, Pi_y)
    axes[2].set_title("1.b Momento conjugado ")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("")

    axes[3].plot(t, E)
    axes[3].set_title("1.b Energía total E(t)")
    axes[3].set_xlabel("t"); axes[3].set_ylabel("E")

    fig.suptitle("Punto 1.b  Problema de Landau (conservación aproximada)")
    fig.savefig("1.b.pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 1.c Sistema binario (dos masas m en G=1)
# Usamos integrador simétrico Velocity-Verlet (leapfrog) para mejor conservación
# Energía total: sum_i (1/2 m v_i^2) - G*m*m/|r1-r2|
# Momento angular total (z): Lz = sum_i m (r_i x v_i)_z
# ============================================================

def velocity_verlet_two_body(r1, v1, r2, v2, m=1.0, G=1.0, dt=1e-3, nsteps=10000):
    #Integra dos cuerpos con interacción gravitacional 1/r con Velocity-Verlet.
    r1 = r1.astype(float).copy()
    r2 = r2.astype(float).copy()
    v1 = v1.astype(float).copy()
    v2 = v2.astype(float).copy()

    # Almacenamiento
    R1 = np.zeros((nsteps+1, 2)); R1[0] = r1
    R2 = np.zeros((nsteps+1, 2)); R2[0] = r2
    V1 = np.zeros((nsteps+1, 2)); V1[0] = v1
    V2 = np.zeros((nsteps+1, 2)); V2[0] = v2

    def accel(r1, r2):
        # a1 = -G*m*(r1-r2)/|r1-r2|^3 ; a2 = -G*m*(r2-r1)/|r2-r1|^3
        dr = r1 - r2
        dist = np.linalg.norm(dr)
        # Evitar singularidad si acaso (no debería ocurrir aquí)
        dist3 = (dist**3 if dist > 1e-12 else 1e-12)
        a1 = -G*m*dr/dist3
        a2 = -G*m*(-dr)/dist3
        return a1, a2

    a1, a2 = accel(r1, r2)

    for i in range(1, nsteps+1):
        # actualizar posiciones
        r1_new = r1 + v1*dt + 0.5*a1*(dt**2)
        r2_new = r2 + v2*dt + 0.5*a2*(dt**2)

        # aceleraciones nuevas
        a1_new, a2_new = accel(r1_new, r2_new)

        # actualizar velocidades
        v1_new = v1 + 0.5*(a1 + a1_new)*dt
        v2_new = v2 + 0.5*(a2 + a2_new)*dt

        # guardar y avanzar
        r1, r2, v1, v2, a1, a2 = r1_new, r2_new, v1_new, v2_new, a1_new, a2_new
        R1[i], R2[i], V1[i], V2[i] = r1, r2, v1, v2

    return R1, V1, R2, V2

def punto_1c():
    # Parámetros problema
    G = 1.0
    m = 1.7
    r1_0 = np.array([0.0, 0.0])
    r2_0 = np.array([1.0, 1.0])
    v1_0 = np.array([0.0, 0.5])
    v2_0 = np.array([0.0,-0.5])
    t0, tf = 0.0, 10.0
    dt = 1e-3
    nsteps = int(np.ceil((tf - t0)/dt))

    R1, V1, R2, V2 = velocity_verlet_two_body(r1_0, v1_0, r2_0, v2_0,m=m, G=G, dt=dt, nsteps=nsteps)

    # Tiempo
    t = np.linspace(t0, tf, nsteps+1)

    # Energía y momento angular totales
    dr = R1 - R2
    dist = np.linalg.norm(dr, axis=1)
    # Potencial gravitacional físico: - G m^2 / r
    U = - G*m*m / np.maximum(dist, 1e-12)
    K = 0.5*m*(np.sum(V1**2, axis=1) + np.sum(V2**2, axis=1))
    E = K + U

    # Lz total
    # r x v en 2D -> z = x*vy - y*vx
    Lz1 = m*(R1[:,0]*V1[:,1] - R1[:,1]*V1[:,0])
    Lz2 = m*(R2[:,0]*V2[:,1] - R2[:,1]*V2[:,0])
    Lz  = Lz1 + Lz2

    # Graficar: x1(t) y x2(t); y1(t) y y2(t); E(t); Lz(t)
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
    axes[0].plot(t, R1[:,0], label="x1")
    axes[0].plot(t, R2[:,0], label="x2", alpha=0.8)
    axes[0].set_title("1.c Binario: x1(t) y x2(t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x"); axes[0].legend()

    axes[1].plot(t, R1[:,1], label="y1")
    axes[1].plot(t, R2[:,1], label="y2", alpha=0.8)
    axes[1].set_title("1.c Binario: y1(t) y y2(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("y"); axes[1].legend()

    axes[2].plot(t, E)
    axes[2].set_title("1.c Energía total E(t)")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E")

    axes[3].plot(t, Lz)
    axes[3].set_title("1.c Momento angular total Lz(t)")
    axes[3].set_xlabel("t"); axes[3].set_ylabel("Lz")

    fig.suptitle("Punto 1.c  Sistema binario (Velocity-Verlet; conservación aproximada)")
    fig.savefig("1.c.pdf", bbox_inches="tight")
    plt.close(fig)