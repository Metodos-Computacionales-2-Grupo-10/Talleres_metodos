import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy import interpolate
import scipy.optimize as opt 
####CARGA DE DATOS#####
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
prominence = 0.05  # para detectar picos (que tal alto debe ser un pico para ser considerado)
radio = 3  # cuántos puntos a cada lado del pico eliminamos

fig, axes = plt.subplots(3, 1, figsize=(8, 10))

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

    # 4. Graficar
    ax.scatter(energia, conteo_original, color='red', alpha=0.4, label='Original')
    ax.scatter(energia[picos], conteo[picos], color='black', s=10, label='Picos')
    ax.scatter(energia_filtrada,conteo_filtrado,label="Sin Picos")
    ax.text(1.03, 0.5,"Anodo de " + nombre, rotation=-90,
        fontsize=12, va='center', ha='right',
        transform=ax.transAxes)
    ax.set_ylabel("Conteo de Fotones")
    ax.set_xlabel("Energía (keV)")
    ax.legend()
plt.tight_layout()
plt.savefig("Taller 1/2a.pdf", bbox_inches="tight", pad_inches=0.1)

'''
Con  los  picos  removidos,  aproxime  la  “barriga”  de  los  espectros  como  mejor  se  le  ocurra;
puede usar interpolación, ajustes a alguna función que se le ocurra, o cualquier método de su
preferencia.
Como antes, grafique un par de espectros y muestre cómo se ajusta su aproximación continua
de la barriga, guarde como  2.b.pdf
'''

### Usamos interpolacion 

fig, axes = plt.subplots(3, 1, figsize=(8, 10))
figura=0

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
    '''HASTA AQUI ES LO MISMO DEL PUNTO ANTERIOR. AHORA INTERPOLA '''
    # 4. Interpolación lineal
    interpolador = interpolate.interp1d(energia_filtrada, conteo_filtrado, kind='linear', fill_value="extrapolate")
    conteo_corregido = interpolador(energia)
    # 5. Graficar
    ax.plot(energia, conteo_original, color='red', alpha=0.4, label='Original')
    ax.plot(energia, conteo_corregido, label='Interpolacion')
    ax.scatter(energia[picos], conteo[picos], color='black',s=10, label='Picos')
    ax.text(1.03, 0.5,"Anodo de "+nombre, rotation=-90,
        fontsize=12, va='center', ha='right',
        transform=ax.transAxes)
    ax.set_ylabel("Conteo de Fotones")
    ax.set_xlabel("Energía (keV)")
    ax.legend()


plt.savefig("Taller 1/2b.pdf", bbox_inches="tight", pad_inches=0.1)

'''
2.c. Analizar el continuo
Con su aproximación, calcule:
• El máximo del continuo (no debe ser un valor del eje 𝑦 exactamente)
• La energía donde ocurre el máximo (no debe ser un número entero)
• El ancho a media altura del continuo, FWHM (no debe ser un número entero)
Grafique estas tres variables como función del voltaje del tubo, y también grafique el máximo
con respecto la energía del máximo. Esto son 4 subplots, donde en cada uno deben estar las
tres gráficas de los tres elementos, con leyenda.
Guarde esta gráfica como  2.c.pdf
'''
###Cree una carga de datos diferente para esta que me pide analizarlos todos
datos={}
materiales_anodo = ["W","Rh","Mo"] #lista materiales
for material in materiales_anodo:
    medidas={}
    for voltaje in range(10,51,1): #rango de 10-50Kv con saltos de 1
        path="Taller 1/"+material+"_unfiltered_10kV-50kV/"+material+"_"+str(voltaje)+"kV.dat" ###Crea el path
        if not os.path.exists(path): 
          print("Archivo no encontrado "+path)  
          continue
        energia, conteo = carga(path)
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
                
        interpolador = interpolate.interp1d(energia_filtrada, conteo_filtrado, kind='linear', fill_value="extrapolate")
        conteo_corregido = interpolador(energia) #carga datos de archivo
        '''Se uso solo interpolacion para la barriga'''
        medidas[str(voltaje)]=pd.DataFrame({"energia": energia,"conteo": conteo_corregido})
    datos[material]=medidas
#Se guardan todos los datos en Df "datos" con llave el elemento y valor otro dict de llave voltaje y valor un df de los datos.
#print(datos['W']['10'][energio o conteo])
valores_grafica3={}
voltajes = list(range(10, 51, 1))
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
##FUNCION QUE PERMITE ENCONTRAR EL ANCHO DE BANDA A ALTURA MEDIA "APROXIMADAMENTE"
def fwhm(y_values, x_values):
    media_altura=max(y_values)/2
    limite_izq=None
    limite_der=None
    for i in range(len(y_values)):
        if y_values[i] >= media_altura:
            limite_izq = x_values[i]
            break

    # Buscar límite derecho
    for i in range(len(y_values) - 1, -1, -1):
        if y_values[i] >= media_altura:
            limite_der = x_values[i]
            break
    return limite_der-limite_izq

for material in materiales_anodo:
    valores_grafica3['xmax_voltaje'] = []
    valores_grafica3['ymax_voltaje'] = []
    valores_grafica3['y_max-x_max'] = []
    valores_grafica3['FWHM_voltaje'] = []

    for voltaje in voltajes:
        y_max = max(datos[material][str(voltaje)]['conteo'])#altura maxima
        indice=list(datos[material][str(voltaje)]['conteo']).index(y_max)#indice en x(energia) donde los conteos son maximos
        x_max=datos[material][str(voltaje)]['energia'][indice] # x(energia) evaluada en el indice
        valores_grafica3['xmax_voltaje'].append((voltaje, x_max))
        valores_grafica3['ymax_voltaje'].append((voltaje, y_max))
        valores_grafica3['y_max-x_max'].append((x_max, y_max))
    # para calcular el ancho de Banda a altura media FWMH
        valores_grafica3['FWHM_voltaje'].append((voltaje,fwhm(datos[material][str(voltaje)]['conteo'],datos[material][str(voltaje)]['energia'])))
    
    # Graficar en cada subplot usando X e Y separados
    axes[0].plot(*zip(*valores_grafica3['xmax_voltaje']), label=material)
    axes[1].plot(*zip(*valores_grafica3['ymax_voltaje']), label=material)
    axes[3].plot(*zip(*valores_grafica3['y_max-x_max']), label=material)

    axes[2].plot(*zip(*valores_grafica3['FWHM_voltaje']),label=material)
# Configuración de ejes y leyendas
axes[0].set_xlabel('Voltaje (kV)')
axes[0].set_ylabel('Energia del Maximo de Conteos')
axes[0].legend()

axes[1].set_xlabel('Voltaje (kV)')
axes[1].set_ylabel('Conteos Maximos')
axes[1].legend()

axes[2].set_xlabel('Voltaje (kV)')
axes[2].set_ylabel('Ancho de Banda a altura media FWHM')
axes[2].legend()

axes[3].set_xlabel('Energia del Maximo de Conteos')
axes[3].set_ylabel('Maximo de Conteos')
axes[3].legend()

plt.tight_layout()
plt.savefig("Taller 1/2c.pdf", bbox_inches="tight", pad_inches=0.1)
