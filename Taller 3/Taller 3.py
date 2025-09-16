


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



# 1.a — Lotka–Volterra


def lv_rhs(t, z, alpha, beta, gamma, delta):
    x, y = z
    dx = alpha*x - beta*x*y
    dy = -gamma*y + delta*x*y
    return [dx, dy]

alpha, beta, gamma, delta = 2.0, 1.5, 0.3, 0.4
x0, y0 = 3.0, 2.0
t0, tf = 0.0, 50.0
t_eval = np.linspace(t0, tf, 10001)

sol = solve_ivp(fun=lambda t, z: lv_rhs(t, z, alpha, beta, gamma, delta),t_span=(t0, tf), y0=[x0, y0],t_eval=t_eval, method="RK45")
t = sol.t
x, y = sol.y
V = delta*x - gamma*np.log(np.maximum(x, 1e-12)) + beta*y - alpha*np.log(np.maximum(y, 1e-12))

fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
axes[0].plot(t, x); axes[0].set_title("1.a Lotka/Volterra: x(t)")
axes[1].plot(t, y); axes[1].set_title("1.a Lotka/Volterra: y(t)")
axes[2].plot(t, V); axes[2].set_title("1.a Cantidad conservada V(t)")
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1a.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)


# 1.b — Landau 2D con E(x), Bz


def landau_rhs(t, y, q, m, E0, B0, k):
    x, y_, vx, vy = y
    Fx = q*E0*(np.sin(k*x) + k*x*np.cos(k*x))
    dvx = (Fx + q*B0*vy)/m
    dvy = (-q*B0*vx)/m
    return [vx, vy, dvx, dvy]

q, m = 7.5284, 3.8428
B0, E0, k = 0.438, 0.7423, 1.0014
x0, y0, vx0, vy0 = 0.1, 0.0, 0.0, 0.2
t0, tf = 0.0, 30.0
t_eval = np.linspace(t0, tf, 60001)

sol = solve_ivp(fun=lambda t, Y: landau_rhs(t, Y, q, m, E0, B0, k),t_span=(t0, tf), y0=[x0, y0, vx0, vy0],t_eval=t_eval, method="RK45")
t = sol.t
x, y, vx, vy = sol.y
K = 0.5*m*(vx**2 + vy**2)
U = - q*E0*x*np.sin(k*x)
E = K + U
Pi_y = m*vy - q*B0*x

fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
axes[0].plot(t, x);    axes[0].set_title("1.b Landau: x(t)")
axes[1].plot(t, y);    axes[1].set_title("1.b Landau: y(t)")
axes[2].plot(t, Pi_y); axes[2].set_title("1.b Momento conjugado Π_y(t)")
axes[3].plot(t, E);    axes[3].set_title("1.b Energía total E(t)")
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1b.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)


# 1.c — Dos cuerpos gravitacionales


def two_body_rhs(t, y, m=1.7, G=1.0):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    dr = r1 - r2
    dist = np.sqrt(dr[0]**2 + dr[1]**2)
    dist3 = dist**3 if dist > 1e-12 else 1e-12
    a1 = -G*m*dr/dist3
    a2 = -G*m*(-dr)/dist3
    return [vx1, vy1, a1[0], a1[1], vx2, vy2, a2[0], a2[1]]

m, G = 1.7, 1.0
r1_0, v1_0 = [0.0, 0.0], [0.0, 0.5]
r2_0, v2_0 = [1.0, 1.0], [0.0,-0.5]
t0, tf = 0.0, 10.0
t_eval = np.linspace(t0, tf, 20001)

y0 = [r1_0[0], r1_0[1], v1_0[0], v1_0[1], r2_0[0], r2_0[1], v2_0[0], v2_0[1]]

sol = solve_ivp(fun=lambda t, y: two_body_rhs(t, y, m=m, G=G),t_span=(t0, tf), y0=y0,t_eval=t_eval, method="RK45")
t = sol.t
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = sol.y
R1 = np.vstack((x1, y1)).T
R2 = np.vstack((x2, y2)).T
V1 = np.vstack((vx1, vy1)).T
V2 = np.vstack((vx2, vy2)).T
dr = R1 - R2
dist = np.linalg.norm(dr, axis=1)
U = - G*m*m / np.maximum(dist, 1e-12)
K = 0.5*m*(np.sum(V1**2, axis=1) + np.sum(V2**2, axis=1))
E = K + U
Lz = m*(R1[:,0]*V1[:,1] - R1[:,1]*V1[:,0]) + m*(R2[:,0]*V2[:,1] - R2[:,1]*V2[:,0])

fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
axes[0].plot(t, x1, label="x1"); axes[0].plot(t, x2, label="x2", alpha=0.85)
axes[1].plot(t, y1, label="y1"); axes[1].plot(t, y2, label="y2", alpha=0.85)
axes[2].plot(t, E);  axes[2].set_title("1.c Energía total E(t)")
axes[3].plot(t, Lz); axes[3].set_title("1.c Momento angular total Lz(t)")
axes[0].set_title("1.c Binario: x1(t), x2(t)"); axes[0].legend()
axes[1].set_title("1.c Binario: y1(t), y2(t)"); axes[1].legend()
for ax in axes: ax.set_xlabel("t")
plt.savefig("Taller 3/1c.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
