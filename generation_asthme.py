import pandas as pd
import numpy as np

# Fixons une graine aléatoire pour la reproductibilité
np.random.seed(42)

# Nombre de lignes
n = 1000

# Génération des données
pm25 = np.random.randint(5, 100, size=n)  # PM2.5 entre 5 et 100
humidite = np.random.randint(30, 90, size=n)  # Humidité entre 30% et 90%
temperature = np.random.randint(10, 35, size=n)  # Température entre 10°C et 35°C

# Probabilité de toux et d'essoufflement augmente avec PM2.5 et humidité
toux = (pm25 + humidite > np.random.randint(50, 150, size=n)).astype(int)
essoufflement = (pm25 + temperature > np.random.randint(50, 150, size=n)).astype(int)

# Probabilité de crise augmente avec PM2.5, toux et essoufflement
crise = ((pm25 > 40) | (toux + essoufflement > 1)).astype(int) * np.random.binomial(1, 0.8, size=n)

# Création du DataFrame
df = pd.DataFrame({
    'PM2.5': pm25,
    'Humidité': humidite,
    'Température': temperature,
    'Toux': toux,
    'Essoufflement': essoufflement,
    'Crise': crise
})

# Sauvegarde du fichier CSV
df.to_csv('données_asthme_test.csv', sep=',', index=False)

print("Fichier CSV généré avec succès : données_asthme_test.csv")