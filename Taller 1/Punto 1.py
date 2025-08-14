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

materiales_anodo = ["W","Rh","Mo"] #lista materiales
fig, axes= plt.subplots(3, 1, figsize=(8, 12)) #grafica de 3 filas 1 columna
figura=0 #numero de fila
for material in materiales_anodo:
    for voltaje in range(10,51,10): #rango de 10-50Kv con saltos de 10
        path="Taller 1/"+material+"_unfiltered_10kV-50kV/"+material+"_"+str(voltaje)+"kV.dat" ###Crea el path
        if not os.path.exists(path): 
          print("Archivo no encontrado "+path)  
          continue
        energia, conteo = carga(path) #carga datos de archivo
        axes[figura].plot(energia,conteo, label=str(voltaje)+"kV")
        axes[figura].legend(title="Voltaje del tubo",loc='upper right')
    axes[figura].set_ylabel("Fluencia")
    axes[figura].text(1.03, 0.5,"Anodo de "+material, rotation=-90,
        fontsize=12, va='center', ha='center',
        transform=axes[figura].transAxes) #titulo ubicado en la derecha rotado 90 grados
    axes[figura].grid(True, which='both', linestyle=':', linewidth=0.5)
    
    axes[figura].set_xlabel("Energía (keV)")
    axes[figura].set_xlim(0, 30) #se limito el rango en X para ver mas claramente los picos caracteristicos  y Bremsstrahlung
    figura+=1

plt.tight_layout()
plt.savefig("Taller 1/1.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 1.pdf")
