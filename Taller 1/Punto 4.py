
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.integrate import simpson

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


materiales_anodo = ["W", "Rh", "Mo"]
prominence = 0.05  # para detectar picos (que tal alto debe ser un pico para ser considerado)
radio = 3  # puntos a cada lado del pico que eliminamos

# Guardar resultados: {material: [(voltaje, porcentaje)]}
resultados = {m: [] for m in materiales_anodo}

for material in materiales_anodo:
    for voltaje in range(10, 50, 1):
        path = f"Taller 1/{material}_unfiltered_10kV-50kV/{material}_{voltaje}kV.dat"
        energia, conteo = carga(path)
        
        # Detectar picos
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
                # Interpolación del continuo
            continuo_interp = np.interp(energia, energia_filtrada, conteo_filtrado)
                
        # Áreas con Simpson
        area_total = simpson(y=conteo, x=energia) #Area total con datos originales
        area_continuo = simpson(y=continuo_interp, x=energia) #continuo (sin picos)
        area_picos = area_total - area_continuo # Área de los picos
            
        # Porcentaje
        porcentaje = (area_picos / area_continuo) * 100 #Que tanto contribuye los picos a comparación del continuo
        resultados[material].append((voltaje, porcentaje))

# Graficar resultado
plt.figure(figsize=(8, 6))
# Recorrer cada material y sus datos
for material, datos in resultados.items():
    # Ordenar la lista 'datos' por el primer elemento de cada tupla (el voltaje)
    datos_ordenados = sorted(datos, key=lambda tupla: tupla[0])  # tupla[0] = voltaje
    # Extraer todos los voltajes en una lista
    voltajes = []
    for tupla in datos_ordenados:
        voltajes.append(tupla[0])
    # Extraer todos los porcentajes en otra lista
    porcentajes = []
    for tupla in datos_ordenados:
        porcentajes.append(tupla[1])
    # Graficar los datos para este material
    plt.plot(voltajes, porcentajes, marker="o", label="Ánodo " + material)


plt.xlabel("Voltaje del tubo (kV)")
plt.ylabel("Área picos / Área continuo (%)")
plt.title("Porcentaje de área de picos vs continuo")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig("Taller 1/4.pdf", bbox_inches="tight", pad_inches=0.1)

