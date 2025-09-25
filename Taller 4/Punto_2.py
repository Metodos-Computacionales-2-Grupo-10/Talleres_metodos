##PUNTO 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
import pde
# Caso Base
a, b = 0.00028, 0.05
eq = pde.PDE(
    {
        "u": f"{a} * laplace(u) + u - u**3 - v - 0.05",
        "v": f"{b} * laplace(v) + 10*(u-v)",
    }
)

# Definiendo el grid (Condiciones iniciales)
grid = pde.CartesianGrid([[0,3], [0,3]], [200,200])
u = pde.ScalarField.random_normal(grid, label="Field $u$")
v = pde.ScalarField.random_normal(grid, label="Field $v$")
state = pde.FieldCollection([u, v])

# Simulación
sol = eq.solve(state, t_range=15, dt=1e-3)
# Caso 1 #Pequeñas formaciones (Mayor ramificación) (separacion entre gotas menor)
a1, b1 = 0.00007, 0.05
eq1 = pde.PDE(
    {
        "u": f"{a1} * laplace(u) + u - u**3 - v - 0.05",
        "v": f"{b1} * laplace(v) + 10*(u-v)",
    }
)

# Definiendo el grid (condiciones iniciales)
grid1 = pde.CartesianGrid([[0,3], [0,3]], [200,200])
u1 = pde.ScalarField.random_normal(grid1, label="Field $u$")
v1 = pde.ScalarField.random_normal(grid1, label="Field $v$")
state1 = pde.FieldCollection([u1, v1])

# Simulacion
sol1 = eq1.solve(state1, t_range=20, dt=1e-3)
# Caso 2 #Mayor tamaño de agrupaciones (Menor ramificacion) (separacion entre gotas mayor)
a2, b2 = 0.0008, 0.05
eq2 = pde.PDE(
    {
        "u": f"{a2} * laplace(u) + u - u**3 - v - 0.05",
        "v": f"{b1} * laplace(v) + 10*(u-v)",
    }
)

# Defieniendo el grid (condiciones iniciales)
grid2 = pde.CartesianGrid([[0,3], [0,3]], [200,200])
u2 = pde.ScalarField.random_normal(grid2, label="Field $u$")
v2 = pde.ScalarField.random_normal(grid2, label="Field $v$")
state1 = pde.FieldCollection([u2, v2])

# Simulacion
sol2 = eq2.solve(state1, t_range=20, dt=1e-3)
x = np.linspace(0, 3, 200)
y = np.linspace(0, 3, 200)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Dependencia del Factor α", fontsize=16)

c1 = axes[0].contourf(X, Y, (sol1[0].data).T, cmap="Blues", levels=100)
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("Pequeñas formaciones (α=0.00007)")

c2 = axes[1].contourf(X, Y, (sol[0].data).T, cmap="Blues", levels=100)
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("Base (α=0.00028)")

c3 = axes[2].contourf(X, Y, (sol2[0].data).T, cmap="Blues", levels=100)
fig.colorbar(c3, ax=axes[2])
axes[2].set_title("Grandes formaciones (α=0.0008)")

fig.text(0.5, 0.01, "Condiciones: β=0.05, F=u-u^3-v-0.05, G=10(u-v). α controla la distancia entre formaciones",
         ha="center", va="bottom", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Taller 4/2_Dependencia_Alpha.png")
# Caso de 1 b
a3, b3 = 0.00028, 0.5
eq3 = pde.PDE(
    {
        "u": f"{a3} * laplace(u) + u - u**3 - v - 0.05",
        "v": f"{b3} * laplace(v) + 10*(u-v)",
    }
)

# Definiendo el grid (Condiciones iniciales)
grid3 = pde.CartesianGrid([[0,3], [0,3]], [200,200])
u3 = pde.ScalarField.random_normal(grid3, label="Field $u$")
v3 = pde.ScalarField.random_normal(grid3, label="Field $v$")
state3 = pde.FieldCollection([u3, v3])

# Simulación
sol3 = eq3.solve(state3, t_range=6, dt=1e-4)
# Caso de 2 b
a4, b4 = 0.00028, 0.015
eq4 = pde.PDE(
    {
        "u": f"{a4} * laplace(u) + u - u**3 - v - 0.05",
        "v": f"{b4} * laplace(v) + 10*(u-v)",
    }
)

# Definiendo el grid (Condiciones iniciales)
grid4 = pde.CartesianGrid([[0,3], [0,3]], [200,200])
u4 = pde.ScalarField.random_normal(grid4, label="Field $u$")
v4 = pde.ScalarField.random_normal(grid4, label="Field $v$")
state4 = pde.FieldCollection([u4, v4])

# Simulación
sol4 = eq4.solve(state4, t_range=15, dt=1e-3)
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("Dependencia del Factor β", fontsize=16)

c11 = axes1[0].contourf(X, Y, (sol3[0].data).T, cmap="BuGn", levels=100)
fig1.colorbar(c11, ax=axes1[0])
axes1[0].set_title("Alta conexion entre formaciones (β=0.5) (dt=1e-4)(t=6)")

c21 = axes1[1].contourf(X, Y, (sol[0].data).T, cmap="BuGn", levels=100)
fig1.colorbar(c21, ax=axes1[1])
axes1[1].set_title("Base (β=0.05)")

c31 = axes1[2].contourf(X, Y, (sol4[0].data).T, cmap="BuGn", levels=100)
fig1.colorbar(c31, ax=axes1[2])
axes1[2].set_title("Poca conexion entre formaciones (β=0.015) (Mayor aislamiento)")

fig1.text(0.5, 0.01, "Condiciones: α=0.00028, F=u-u^3-v-0.05, G=10(u-v). β controla la conexion entre formaciones y la regularidad de estas",
         ha="center", va="bottom", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Taller 4/2_Dependencia_Beta.png")
# Caso de Funciones 1
eq5 = pde.PDE(
    {
        "u": f"{a} * laplace(u) + u - u**3 - v- 0.05",
        "v": f"{b} * laplace(v) + 0.01*(u-v)",
    }
)

eq6 = pde.PDE(
    {
        "u": f"{a} * laplace(u) + u - u**3 - v- 0.05",
        "v": f"{b} * laplace(v) + 0.1*(u-v)",
    }
)

eq7 = pde.PDE(
    {
        "u": f"{a} * laplace(u) + u - u**3 - v- 0.05",
        "v": f"{b} * laplace(v) + 1*(u-v)",
    }
)

sol5 = eq5.solve(state, t_range=15, dt=1e-3)
sol6 = eq6.solve(state, t_range=15, dt=1e-3)
sol7 = eq7.solve(state, t_range=15, dt=1e-3)
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("Dependencia del numero en R(u-v)", fontsize=16)

c12 = axes2[0].contourf(X, Y, (sol5[0].data).T, cmap="PuOr", levels=100)
fig2.colorbar(c12, ax=axes2[0])
axes2[0].set_title("Muy poca probabilidad de formaciones (moradas) (R=0.01)")

c22 = axes2[1].contourf(X, Y, (sol6[0].data).T, cmap="PuOr", levels=100)
fig2.colorbar(c22, ax=axes2[1])
axes2[1].set_title("Poca probabilidad de formaciones (moradas) (R=0.1)")

c32 = axes2[2].contourf(X, Y, (sol7[0].data).T, cmap="PuOr", levels=100)
fig2.colorbar(c32, ax=axes2[2])
axes2[2].set_title("Mediana probabilidad de formaciones (moradas) (R=1)")

c42 = axes2[3].contourf(X, Y, (sol[0].data).T, cmap="PuOr", levels=100)
fig2.colorbar(c42, ax=axes2[3])
axes2[3].set_title("Normal probabilidad de formaciones (moradas) (R=10)")


fig2.text(0.5, 0.01, "Condiciones: α=0.00028, β=0.05, F=u-u^3-v-0.05, G=R(u-v). R controla Recurrencia de formaciones elevadas (en morado)",
         ha="center", va="bottom", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Taller 4/2_Dependencia_R(Factor R(u-v)).png")
plt.close()

eq10 = pde.PDE(
    {
        "u": f"{a} * laplace(u) + u - v*u**2",
        "v": f"{b} * laplace(v) + 10*(u-v)",
    }
)


sol10=eq10.solve(state, t_range=15, dt=1e-3)
plt.contourf(X,Y,(sol10[0].data).T, cmap="ocean", levels=100)
plt.title("Algas (Función 1)")
plt.colorbar()
plt.xlabel("Condiciones: α=0.00028, β=0.05, F=u-v*u**2, G=10(u-v). Formación ramificada similar a algas", fontsize=7)
plt.savefig("Taller 4/2_Algas.png")
plt.close()


a18, dx8, dy8 = 0.05, 0.2, 0.2
eq18 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.025",
        "v": f"{dx8}*d2_dx2(v) + {dy8}*d2_dy2(v) - v*u**2 + 5",
    }
)

sol18=eq18.solve(state, t_range=15, dt=0.5e-4)

plt.contourf(X,Y,(sol18[1].data).T, cmap="copper", levels=100)
plt.title("Jaguar (S=0.05, T=5, dx=dy=0.2)")
plt.xlabel("Condiciones:Ut= 0.0005*∇*2 u+v*u2-u-0.025, Vt=0.2*d2_dx2(v)+0.2*d2_dy2(v)-v*u*2+5", fontsize=7)
plt.colorbar()
plt.savefig("Taller 4/2_Jaguar.png")
plt.close()

a12, dx2, dy2 = 0.0005, 0.01, 0.02
eq12 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.025",
        "v": f"{dx2}*d2_dx2(v) + {dy2}*d2_dy2(v) - v*u**2 + 1.55",
    }
)

sol12=eq12.solve(state, t_range=50, dt=1e-3)

dx3, dy3 = 0.02, 0.02
eq13 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.025",
        "v": f"{dx3}*d2_dx2(v) + {dy3}*d2_dy2(v) - v*u**2 + 1.55",
    }
)

sol13=eq13.solve(state, t_range=50, dt=1e-3)


dx4, dy4 = 0.02, 0.01
eq14 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.025",
        "v": f"{dx4}*d2_dx2(v) + {dy4}*d2_dy2(v) - v*u**2 + 1.55",
    }
)

sol14=eq14.solve(state, t_range=50, dt=1e-3)

fig4, axes4= plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle("Dependencia de dx y dy (lineas y puntos)", fontsize=16)

b1=axes4[0].contourf(X, Y, (sol12[0].data).T, cmap="afmhot", levels=100)
fig4.colorbar(b1, ax=axes4[0])
axes4[0].set_title("dx=0.01, dy=0.02, dx/dy=1/2")

b2=axes4[1].contourf(X, Y, (sol13[0].data).T, cmap="afmhot", levels=100)
fig4.colorbar(b2, ax=axes4[1])
axes4[1].set_title("dx=0.02, dy=0.02, dx7dy=1")

b3=axes4[2].contourf(X, Y, (sol14[0].data).T, cmap="afmhot", levels=100)
fig4.colorbar(b3, ax=axes4[2])
axes4[2].set_title("dx=0.02, dy=0.01, dx/dy=2")

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig4.text(0.5, 0.01, "Condiciones:Ut= 0.0005*∇*2 u+v*u2-u-0.025, Vt=dx*d2_dx2(v)+dy*d2_dy2(v)-v*u*2+1.55",
         ha="center", va="bottom", fontsize=12)
plt.savefig("Taller 4/2_Dependencia_dx_dy.png")

a11, dx, dy = 0.005, 0.01, 0.01
eq11 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.3",
        "v": f"{dx}*d2_dx2(v) + {dy}*d2_dy2(v) - v*u**2 + 3",
    }
)

sol11=eq11.solve(state, t_range=20, dt=1e-3)

a17, dx7, dy7 = 0.005, 0.01, 0.01
eq17 = pde.PDE(
    {
        "u": f"0.0005*laplace(u) + v*u**2 - u + 0.025",
        "v": f"{dx7}*d2_dx2(v) + {dy7}*d2_dy2(v) - v*u**2 + 5",
    }
)

sol17=eq17.solve(state, t_range=7, dt=0.5e-3)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))
fig3.suptitle("Diferentes patrones con la función 2", fontsize=16)

a3 = axes3[0].contourf(X, Y, (sol11[0].data).T, cmap="berlin", levels=100)
fig3.colorbar(a3, ax=axes3[0])
axes3[0].set_title("Piel de pez loro (S=0.3, T=3)")

a32 = axes3[1].contourf(X, Y, (sol17[0].data).T, cmap="berlin", levels=100)
fig3.colorbar(a32, ax=axes3[1])
axes3[1].set_title("Estructura de red (S=0.025, T=5, t=7)")

fig3.text(0.5, 0.01, "Condiciones:Ut= 0.0005*∇*2 u+v*u2-u-S, Vt=0.01*d2_dx2(v)+0.01*d2_dy2(v)-v*u*2+T",
         ha="center", va="bottom", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Taller 4/2_Pez_loro_y_red.png")
