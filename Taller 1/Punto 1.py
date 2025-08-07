import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.signal import find_peaks
from scipy import interpolate
'''
1. CARGA DATOS y grafica
Importe los datos y diseñe una manera de visualizar todas las variables (energía, conteo de
fotones,  kilovoltaje  del  tubo  y  elemento  del  ánodo)  en  una  gráfica,  como  si  fuera  para  un
artículo científico.
Puede usar cualquier tipo de gráfico. Puede usar color. Puede usar subplots. No tiene que hacer
uso de todos los datos, pero se debe poder visualizar el impacto de cada variable. Se calificará
creatividad y visibilidad.
Guarde su gráfica con el nombre  1.pdf .
'''
def carga(path):
    energia = []  # Lista para guardar energías
    conteo = []  # Lista para guardar conteos

    with open(path, 'r', encoding='latin1') as file: ### No dejaba con utf-8(el estandar) Chat GPT nos recomendo usar encoding='latin1"
        for linea in file:
            linea = linea.strip()
            if linea == '' or linea.startswith('#'):
                continue
            parts = linea.split()
            if len(parts) >= 2:
                    e = float(parts[0])     # energía
                    c = float(parts[1])     # conteo
                    energia.append(e)        # guardamos energía
                    conteo.append(c)        # guardamos conteo
    return np.array(energia), np.array(conteo)

materiales_anodo = ["W","Rh","Mo"]
fig, axes= plt.subplots(3, 1, figsize=(8, 12)) #grafica de 3 filas 1 columna
figura=0 #numero de fila
for material in materiales_anodo:
    for voltaje in range(10,60,10):
        path="Taller 1/"+material+"_unfiltered_10kV-50kV/"+material+"_"+str(voltaje)+"kV.dat" ###Crea el path
        if not os.path.exists(path): 
          print("Archivo no encontrado "+path)  
          continue
        energia, conteo = carga(path)
        axes[figura].plot(energia,conteo, label=str(voltaje)+"kV")
        axes[figura].legend(title="Voltaje del tubo",loc='upper right')
    axes[figura].set_ylabel("Conteo de fotones")
    axes[figura].text(1.03, 0.5,"Anodo de "+material, rotation=-90,
        fontsize=12, va='center', ha='center',
        transform=axes[figura].transAxes)
    axes[figura].grid(True, which='both', linestyle=':', linewidth=0.5)
    
    axes[figura].set_xlabel("Energía (keV)")
    axes[figura].set_xlim(0, 30)
    figura+=1

plt.tight_layout()
plt.savefig("Taller 1/1.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 1.pdf")
# Crear gráfico

'''
2. Comportamiento del continuo (Bremsstrahlung) 
2.a. Remover los picos
Cree  una  copia  de  los  datos  originales,  y
procéselos  de  tal  manera  que  los  puntos  co,
rrespondientes a los picos queden eliminados.
Haga algunos (dos o tres) subplots con ejem,
plos  distintos  de  los  picos  removidos  como
se muestra en la imagen, pero no necesaria,
mente en el mismo estilo o de la misma forma.
Guarde esta gráfica como  2.a.pdf
'''

archivos = {
    "Tungsteno (W)": "Taller 1/W_unfiltered_10kV-50kV/W_30kV.dat",
    "Rodio (Rh)": "Taller 1/Rh_unfiltered_10kV-50kV/Rh_30kV.dat",
    "Molibdeno (Mo)": "Taller 1/Mo_unfiltered_10kV-50kV/Mo_30kV.dat"
}

# Parámetros
prominence = 0.05
radio = 3  # cuántos puntos a cada lado del pico eliminamos

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

for ax, (nombre, archivo) in zip(axes, archivos.items()):
    energia, conteo = carga(archivo)
    conteo_original = conteo.copy()

    # 1. Detectar picos
    altura_minima = prominence * np.max(conteo)
    picos, _ = find_peaks(conteo, prominence=altura_minima)

    # 2. Obtener índices a eliminar (alrededor de cada pico)
    indices_a_eliminar = set()
    for pico in picos:
        for i in range(pico - radio, pico + radio):
            indices_a_eliminar.add(i)

    # 3. Crear nuevos arrays solo con datos válidos 
    energia_filtrada = []
    conteo_filtrado = []
    for i in range(len(conteo)):
        if i not in indices_a_eliminar:
            energia_filtrada.append(energia[i])
            conteo_filtrado.append(conteo[i])

    # 4. Interpolación lineal
    interpolador = interpolate.interp1d(energia_filtrada, conteo_filtrado, kind='linear', fill_value="extrapolate")
    conteo_corregido = interpolador(energia)

    # 5. Graficar
    ax.plot(energia, conteo_original, color='red', alpha=0.4, label='Original')
    ax.plot(energia, conteo_corregido, label='Corregido')
    ax.scatter(energia[picos], conteo[picos], color='black', s=10, label='Picos')
    ax.set_title(nombre)
    ax.set_ylabel("Cuentas")
    ax.legend()

axes[-1].set_xlabel("Energía (keV)")
plt.tight_layout()
plt.savefig("Taller 1/2a.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 2a.pdf")