import numpy as np
from matplotlib import pyplot as plt
import os

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

# Lista explícita de los 15 archivos
archivos = {
    "W": [
        "W_unfiltered_10kV-50kV/W_10kV.dat",
        "W_unfiltered_10kV-50kV/W_20kV.dat",
        "W_unfiltered_10kV-50kV/W_30kV.dat",
        "W_unfiltered_10kV-50kV/W_40kV.dat",
        "W_unfiltered_10kV-50kV/W_50kV.dat"
    ],
    "Rh": [
        "Rh_unfiltered_10kV-50kV/Rh_10kV.dat",
        "Rh_unfiltered_10kV-50kV/Rh_20kV.dat",
        "Rh_unfiltered_10kV-50kV/Rh_30kV.dat",
        "Rh_unfiltered_10kV-50kV/Rh_40kV.dat",
        "Rh_unfiltered_10kV-50kV/Rh_50kV.dat"
    ],
    "Mo": [
        "Mo_unfiltered_10kV-50kV/Mo_10kV.dat",
        "Mo_unfiltered_10kV-50kV/Mo_20kV.dat",
        "Mo_unfiltered_10kV-50kV/Mo_30kV.dat",
        "Mo_unfiltered_10kV-50kV/Mo_40kV.dat",
        "Mo_unfiltered_10kV-50kV/Mo_50kV.dat"
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
plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
print("Gráfico guardado como 1.pdf")