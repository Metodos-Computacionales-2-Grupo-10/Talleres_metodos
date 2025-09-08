import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

"""EJERCICIO 2 BALISTICA"""
m=10.01 #Kg
pasos=1000
v_0=np.linspace(0,140,pasos) #en m/s
theta=np.linspace(0,80,pasos) #En grados
g=9.773 #m/s^2
y_0=0.
v=v_0*np.cos(np.radians(theta)), v_0*np.sin(np.radians(theta))
estado_inicial=[0.,y_0,v[0][0],v[1][0]] #x0,y0,vx0,vy0
##Creamos una funcionn que define la friccion del aire
def f_friccion(y,v,A=1.642,B=40.624,C=2.36):
    coef_friccion=A*(1-y/B)**C
    return (-coef_friccion*np.linalg.norm(v)**2, -coef_friccion*np.linalg.norm(v)**2)
'''2.a ALCANCE
Queremos encontrar el alcance máximo horizontal como función de la velocidad inicial. Si
no hubiese fricción con el aire, sería 45° sin importar la velocidad. ¿Qué comportamiento se
observa en este caso?
Guarde su gráfica de 𝑥max vs 𝑣0 en  2.a.pdf'''



'''2.a.1 BONO
Si 𝑣0 no es un número, sino una lista/array/tupla de dos valores, considere éstos como 𝑣0 ± 𝜎𝑣0
y propague el error a su respuesta.'''



'''2.b Atinar a un objetivo
Escriba una función  angle_to_hit_target(v0,target_x,target_y)  que dada una veloci0
dad inicial fija, encuentre el ángulo necesario 𝜃0 para que el cañón atine a un objetivo en las
coordenadas  target_x,target_y , ambas positivas.
Debería probar con varios objetivos, graficar la trayectoria y asegurarse de que su código
funciona, pues no queremos decepcionar al general. No necesita guardar estas gráficas.
'''

'''2.c Varias opciones para disparar
Dado un objetivo (𝑥, 𝑦) existen varias combinaciones de 𝑣0 y 𝜃0 que hacen que el disparo
llegue al objetivo. Caracterice este conjunto de soluciones. ¿Qué dimensión tiene? ¿Se le ocurre
alguna parametrización?
Cree una gráfica ( 2.c.pdf ) de 𝜃0 vs 𝑣0 que muestre dónde están las condiciones iniciales que
atinan al objetivo en (𝑥, 𝑦) = (12m, 0)
'''