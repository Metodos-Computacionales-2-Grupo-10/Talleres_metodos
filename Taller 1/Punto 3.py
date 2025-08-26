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
<<<<<<< HEAD
import re #validar, extraer o buscar patrones de texto
def carga(path):
    energia = []  # Lista para guardar energías
    conteo = []  # Lista para guardar conteos

    with open(path, 'r', encoding='latin1') as file: ### No dejaba con utf-8(el estandar) Chat GPT nos recomendo usar encoding='latin1"
=======


def carga(path):
    energia = []  # Lista para guardar energías
    conteo = []   # Lista para guardar conteos
    with open(path, 'r', encoding='latin1') as file: #abrir el archivo, el encoding ignora lineas vacias y numerales, lee dos columnas
>>>>>>> main
        for linea in file:
            linea = linea.strip()
            if linea == '' or linea.startswith('#'):
                continue
            parts = linea.split()
            if len(parts) >= 2:
<<<<<<< HEAD
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
=======
                e = float(parts[0])     # energía
                c = float(parts[1])     # conteo
                energia.append(e)
                conteo.append(c)
    return np.array(energia), np.array(conteo) #retorna estos dos arrays con la energia y el conteo

prominence = 0.05         # fracción del máximo para detectar (que tan alto debe ser un pico para ser considerado)
radio = 3                 # cuántos puntos a cada lado del pico eliminamos
materiales_anodo = ["W","Rh","Mo"]
voltajes = list(range(10, 51, 1))  # 10–50 kV con saltos de uno en uno 


def continuo_interpolado(energia, conteo, prominence, radio):
    
    altura_minima = prominence * np.max(conteo) if np.max(conteo) > 0 else 0.0
    picos, _ = find_peaks(conteo, prominence=altura_minima)

    indices_a_eliminar = set()
    for pico in picos:
        for i in range(pico - radio, pico + radio + 1):  # incluir el extremo derecho
            if 0 <= i < len(conteo):
                indices_a_eliminar.add(i)

    energia_filtrada = []
    conteo_filtrado = []
    for i in range(len(conteo)):
        if i not in indices_a_eliminar:
            energia_filtrada.append(energia[i])
            conteo_filtrado.append(conteo[i])

    if len(energia_filtrada) < 2:
        # si no hay suficientes puntos para interpolar, devolvemos el original
        return conteo.copy()

    interpolador = interpolate.interp1d(np.array(energia_filtrada), np.array(conteo_filtrado),kind='linear', fill_value="extrapolate")
    return interpolador(energia)




fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
titulos = {"Mo": "Molibdeno (Mo)", "Rh": "Rodio (Rh)", "W": "Tungsteno (W)"}
orden_materiales = ["Mo", "Rh", "W"]  # orden visual

# También almacenamos y_peaks para reutilizar en el siguiente punto 
y_peaks_dict = {m: {} for m in materiales_anodo}
x_dict       = {m: {} for m in materiales_anodo}

for ax, mat in zip(axes, orden_materiales):
    base_dir = f"Taller 1/{mat}_unfiltered_10kV-50kV"
    if not os.path.isdir(base_dir):
        ax.set_title(f"{titulos.get(mat, mat)} (sin datos)")
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel("Intensidad (picos)")
        continue

    for v in voltajes:
        path = os.path.join(base_dir, f"{mat}_{v}kV.dat")
        if not os.path.exists(path):
            continue

        energia, conteo = carga(path)
        # asegurar orden creciente en energía
        idx = np.argsort(energia)
        energia = energia[idx]
        conteo  = conteo[idx]

        
        conteo_cont = continuo_interpolado(energia, conteo, prominence, radio)
        # Sacamos solo los picos 
        y_peaks = conteo - conteo_cont

        # guardar para 3.b
        x_dict[mat][v] = energia
        y_peaks_dict[mat][v] = y_peaks

        # graficar
        ax.plot(energia, y_peaks, lw=1.0, label=f"{v} kV")
        

    ax.set_title(titulos.get(mat, mat))
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel("Intensidad (picos)")
    ax.legend(ncol=2, fontsize=8)
    ax.set_xlim(8,50)

plt.tight_layout()
plt.savefig("Taller 1/3a.pdf", bbox_inches="tight", pad_inches=0.1)
plt.xlim(1,50) #Estos limites son solo para las graficas del Tungsteno
plt.close()
>>>>>>> main
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
<<<<<<< HEAD
(Puede omitir algunos de los espectros iniciales que no presentan picos)"""
=======
(Puede omitir algunos de los espectros iniciales que no presentan picos)"""

def gaussiana(x, media, desviacion, amplitud):
    return amplitud*np.exp((-(x - media)*2) / (2 * desviacion*2)) / (desviacion * np.sqrt(2 * np.pi))

def fwhm_from_sigma(sigma):
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

def ajustar_pico_mayor(x, y_peaks):
    #Busca el pico más alto en y_peaks y ajusta una gaussiana local retorna (altura_ajustada, media, fwhm).


    # detectar picos en y_peaks para ubicar el principal
    rng = y_peaks.max() - y_peaks.min()
    prom = max(0.05 * rng, 1e-12)
    pk, _ = find_peaks(y_peaks, prominence=prom)
    if len(pk) == 0:
        return None
    p_main = pk[np.argmax(y_peaks[pk])]

    # Ventana local alrededor del pico
    w = max(10, int(0.01 * len(x)))
    li = max(p_main - w, 0)
    ri = min(p_main + w, len(x) - 1)

    x_win = x[li:ri+1]
    y_win = y_peaks[li:ri+1]

    # Iniciales para el ajuste
    media0 = x[p_main]
    dx = np.median(np.diff(x_win)) if len(x_win) > 1 else 1.0
    desviacion0 = max(3*dx, dx)
    amplitud0 = max(y_win.max(), 1e-6)  # amplitud del gaussiano

    p0 = [media0, desviacion0, amplitud0]

    try:
        popt, _ = scipy.optimize.curve_fit(gaussiana, x_win, y_win, p0=p0, maxfev=10000)
        media, desviacion, amplitud = popt
        fwhm = fwhm_from_sigma(abs(desviacion))
        altura = gaussiana(media, media, abs(desviacion), amplitud)  # altura en el centro
        return float(altura), float(media), float(fwhm)
    except Exception:
        return None

# Recopilar resultados para graficar
rows_altura = []
rows_fwhm   = []

for mat in materiales_anodo:
    for v in voltajes:
        if v not in x_dict.get(mat, {}) or v not in y_peaks_dict.get(mat, {}):
            continue
        x = x_dict[mat][v]
        y_peaks = y_peaks_dict[mat][v]
        res = ajustar_pico_mayor(x, y_peaks)
        if res is None:
            continue
        altura, media, fwhm = res
        rows_altura.append((mat, v, altura))
        rows_fwhm.append((mat, v, fwhm))

# Graficar Altura y FWHM vs kV (dos subplots)
fig, axes = plt.subplots(2, 1, figsize=(7, 8))

# Altura vs kV
for mat in materiales_anodo:
    pares = [(v, a) for (m, v, a) in rows_altura if m == mat]
    if not pares:
        continue
    pares.sort(key=lambda t: t[0])
    axes[0].plot([p[0] for p in pares], [p[1] for p in pares], marker='o', lw=1.0, label=mat)
axes[0].set_xlabel("kV")
axes[0].set_ylabel("Altura del pico (Gauss)")
axes[0].set_title("Altura del pico mayor vs kV")
axes[0].legend()

# FWHM vs kV
for mat in materiales_anodo:
    pares = [(v, f) for (m, v, f) in rows_fwhm if m == mat]
    if not pares:
        continue
    pares.sort(key=lambda t: t[0])
    axes[1].plot([p[0] for p in pares], [p[1] for p in pares], marker='s', lw=1.0, label=mat)
axes[1].set_xlabel("kV")
axes[1].set_ylabel("FWHM (unidades de energía)")
axes[1].set_title("FWHM del pico mayor vs kV")
axes[1].legend()

plt.tight_layout()
plt.savefig("Taller 1/3b.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close()
>>>>>>> main
