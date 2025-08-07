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

    with open(path, 'r', encoding='latin1') as file:
        for linea in file:
            linea = linea.strip()
            if linea == '' or linea.startswith('#'):
                continue
            parts = linea.split()
            if len(parts) >= 2:
                try:
                    e = float(parts[0])     # energía
                    c = float(parts[1])     # conteo
                    energia.append(e)        # guardamos energía
                    conteo.append(c)        # guardamos conteo
                except:
                    continue
    return np.array(energia), np.array(conteo)

# Lista Archivos
archivos = {
    "W": [
        "Taller 1/W_unfiltered_10kV-50kV/W_10kV.dat",
        "Taller 1/W_unfiltered_10kV-50kV/W_20kV.dat",
        "Taller 1/W_unfiltered_10kV-50kV/W_30kV.dat",
        "Taller 1/W_unfiltered_10kV-50kV/W_40kV.dat",
        "Taller 1/W_unfiltered_10kV-50kV/W_50kV.dat"
    ],
    "Rh": [
        "Taller 1/Rh_unfiltered_10kV-50kV/Rh_10kV.dat",
        "Taller 1/Rh_unfiltered_10kV-50kV/Rh_20kV.dat",
        "Taller 1/Rh_unfiltered_10kV-50kV/Rh_30kV.dat",
        "Taller 1/Rh_unfiltered_10kV-50kV/Rh_40kV.dat",
        "Taller 1/Rh_unfiltered_10kV-50kV/Rh_50kV.dat"
    ],
    "Mo": [
        "Taller 1/Mo_unfiltered_10kV-50kV/Mo_10kV.dat",
        "Taller 1/Mo_unfiltered_10kV-50kV/Mo_20kV.dat",
        "Taller 1/Mo_unfiltered_10kV-50kV/Mo_30kV.dat",
        "Taller 1/Mo_unfiltered_10kV-50kV/Mo_40kV.dat",
        "Taller 1/Mo_unfiltered_10kV-50kV/Mo_50kV.dat"
    ]
}
# Crear gráfico
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

for ax, (element, file_list) in zip(axes, archivos.items()):
    for file in file_list:
        if not os.path.exists(file):
            print(f"Archivo no encontrado: {file}")
            continue
        kv = file.split("_")[-1].replace("kV.dat", "") + " kV"
        energia, conteo = carga(file)
        if energia is not None:
            ax.plot(energia, conteo, label=kv)
    ax.set_yscale('log')
    ax.set_ylabel("Conteo de fotones")
    ax.set_title(f"Anodo de {element}")
    ax.legend(title="Voltaje del tubo")
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

axes[-1].set_xlabel("Energía (keV)")
plt.tight_layout()
plt.savefig("Taller 1/1.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 1.pdf")
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
    "Tungsteno (W)": "Taller 1/W_unfiltered_10kV-50kV/W_40kV.dat",
    "Rodio (Rh)": "Taller 1/Rh_unfiltered_10kV-50kV/Rh_40kV.dat",
    "Molibdeno (Mo)": "Taller 1/Mo_unfiltered_10kV-50kV/Mo_40kV.dat"
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
    ax.set_yscale('log')
    ax.set_ylabel("Cuentas")
    ax.legend()

axes[-1].set_xlabel("Energía (keV)")
plt.tight_layout()
plt.savefig("Taller 1/2a.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 2a.pdf")