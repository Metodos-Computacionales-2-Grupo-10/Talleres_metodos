'''3. Picos (Rayos-X característicos) [2.5pt]

3.a. Aislar los picos
Reste el continuo hallado en el punto anterior
al espectro original para obtener, ojalá, sólo
los picos.
Grafique sólo los picos de todos los espectros,
todos  en  la  misma  gráfica,  para  intentar  ob,
servar si cambian. Esto para cada elemento, es
decir, 3 subplots con título. No queremos ver
todo el espectro completo: haga un zoom en
el eje 𝑥 para mostrar sólo las inmediaciones
de los picos. Guardar en  3.a.pdf
'''
'''3. Picos (Rayos-X característicos) [2.5pt]

3.a. Aislar los picos
Reste el continuo hallado en el punto anterior
al espectro original para obtener, ojalá, sólo
los picos.
Grafique sólo los picos de todos los espectros,
todos  en  la  misma  gráfica,  para  intentar  ob,
servar si cambian. Esto para cada elemento, es
decir, 3 subplots con título. No queremos ver
todo el espectro completo: haga un zoom en
el eje 𝑥 para mostrar sólo las inmediaciones
de los picos. Guardar en  3.a.pdf
'''
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.signal import find_peaks
from scipy import interpolate
import scipy
import re #validar, extraer o buscar patrones de texto
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
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def gaussiana(x, A, mu, sigma):
    return A * np.exp(-((x - mu)*2) / (2 * sigma*2))
def fwhm_from_sigma(sigma):
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
prominence_frac=0.05 #Fracción de la altura máxima para detectar los picos
radio = 3 # cuántos puntos a cada lado del pico eliminamos
base_dir = "Taller 1" #Guarda la ruta base donde están los datos y se guardan las salidas
salida_3a = os.path.join(base_dir, "3a.pdf")
salida_3b = os.path.join(base_dir, "3b.pdf")
#Recorrer absolutamente todos los archivos, por eso no se utiliza el diccionario archivos como en el punto 2
carpetas = {
    "Mo": os.path.join(base_dir, "Mo_unfiltered_10kV-50kV"),
    "Rh": os.path.join(base_dir, "Rh_unfiltered_10kV-50kV"),
    "W" : os.path.join(base_dir, "W_unfiltered_10kV-50kV"),
}

# data[elemento][kv] = (x, y)
data = {"Mo": {}, "Rh": {}, "W": {}}

for elem, carpeta in carpetas.items():
    if not os.path.isdir(carpeta):
        continue
    for fn in sorted(os.listdir(carpeta)):
        if not fn.lower().endswith((".txt", ".dat", ".csv")):
            continue
        ruta = os.path.join(carpeta, fn)
        # Inferir kV del nombre de archivo: p.ej. "Mo_30kV.dat"
        m = re.search(r'(\d{2,3})\s*(kV|kv)', fn) #Esta linea de codigo rara es el patrón 
        if not m:
            # Intentar con la ruta completa por si el nombre no contiene el patrón
            m = re.search(r'(\d{2,3})\s*(kV|kv)', ruta, re.IGNORECASE)
        if not m:
            continue
        kv = int(m.group(1))

        try:
            x, y = carga(ruta)
        except Exception: #si ocurre cualquier error durante la ejecucion de la preuab try, se atrapa y permite que el codigo siga corriendo
            continue

        if len(x) < 10 or len(y) < 10:
            continue

        # Ordenar por x por si acaso
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        data[elem][kv] = (x, y)

# Validación mínima
data = {k: v for k, v in data.items() if len(v) > 0}
assert len(data) > 0, "No se cargaron espectros válidos desde las carpetas descomprimidas."

def construir_continuo_por_interpolacion_lineal(x, y, prominence_frac=0.05, radio=3):
   
    if len(x) < 3:
        return y.copy()

    altura_minima = prominence_frac * np.max(y) if np.max(y) > 0 else 0.0
    picos, _ = find_peaks(y, prominence=altura_minima) #devuelve los indices donde hay picos suficientemnete prominentes

    # Índices a eliminar
    indices_a_eliminar = set()
    for pico in picos:
        for i in range(pico - radio, pico + radio + 1): #por cada pico acotamos y eliminamos 
            if 0 <= i < len(y):
                indices_a_eliminar.add(i)

    # Puntos válidos
    x_fil, y_fil = [], []
    for i in range(len(y)):
        if i not in indices_a_eliminar:
            x_fil.append(x[i])
            y_fil.append(y[i])

    if len(x_fil) < 2:
        # Si no hay suficientes puntos para interpolar, devolver y
        return y.copy()

    interpolador = interpolate.interp1d(
        np.array(x_fil), np.array(y_fil),
        kind='linear', fill_value="extrapolate"
    )
    y_continuo = interpolador(x)
    return y_continuo #se evalua ese interpolador en todas las x originales

ensure_dir(base_dir)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
elem_list = ["Mo", "Rh", "W"]
titulos = {"Mo": "Molibdeno (Mo)", "Rh": "Rodio (Rh)", "W": "Tungsteno (W)"}

for ax, elem in zip(axes, elem_list):
    if elem not in data or len(data[elem]) == 0:
        ax.set_title(f"{titulos.get(elem, elem)} (sin datos)")
        ax.set_xlabel("Energía / Longitud de onda")
        ax.set_ylabel("Intensidad (solo picos)")
        continue

    all_peak_xmin, all_peak_xmax = [], []

    for kv in sorted(data[elem].keys()):
        x, y = data[elem][kv]
        y_cont = construir_continuo_por_interpolacion_lineal(x, y, prominence_frac, radio)
        y_peaks = y - y_cont #aislar solo los picos

        # Ventana de picos para zoom automático
        rng = y_peaks.max() - y_peaks.min()
        prom = max(0.05 * rng, 1e-12)
        pk, _ = find_peaks(y_peaks, prominence=prom)
        if len(pk) > 0:
            px = x[pk]
            all_peak_xmin.append(px.min())
            all_peak_xmax.append(px.max())

        ax.plot(x, y_peaks, lw=1.0, alpha=0.9, label=f"{kv} kV")

    if len(all_peak_xmin) > 0:
        xmin = min(all_peak_xmin)
        xmax = max(all_peak_xmax)
        dx = (xmax - xmin) * 0.15 if xmax > xmin else 1.0
        ax.set_xlim(xmin - dx, xmax + dx)

    ax.set_title(titulos.get(elem, elem))
    ax.set_xlabel("Energía (keV) / Longitud de onda (pm)")
    ax.set_ylabel("Intensidad (picos)")
    ax.legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig(salida_3a, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
print("3.a) Guardado:", salida_3a)
#Eliminar localmente los picos, reconstruir el continuo con una recta entre los puntos vecinos y restarlo al original
#Al restar, el continuo desaparece, idealmente, y quedan solo los picos, por eso las curvas resultantes muestran agudos alrededor de las energías    
"""
de los picos. Guardar en  3.a.pdf
3.b. Ajustar
Para  el  mayor  pico  de  cada  espectro,  para  cada  espectro,  para  cada  elemento,  ajuste  una
función. Aquí no se vale interpolar, porque como podrá notar, los picos están mal muestrados,
teniendo uno o dos puntos de datos.
Se recomienda un modelo Gaussiano, pero puede usar el que desee, con tal que pueda calular
la altura, el ancho a media altura, y la posición de cada pico.
Grafique la altura del pico y el ancho a media altura en función del voltaje del tubo. Estos dos
subplots con los resultados para todos los elementos. Guarde en  3.b.pdf
(Puede omitir algunos de los espectros iniciales que no presentan picos)"""