"""
François Deslandes, Agathe Leschiera, Solène Von Mullum
Prediction d'asthme par machine learning
03/12/2025
"""


# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Chargement des données
df = pd.read_csv('asthme.csv', sep=';') 


# Préparation des données
# X = variables d'entrée (features)
X = df[['PM2.5', 'Humidite', 'Temperature', 'Toux', 'Essoufflement']]
# y = variable à prédire (target)
y = df['Crise']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Création et entraînement du modèle
# Créer un modèle d'arbre de décision
model = DecisionTreeClassifier(max_depth=6,random_state=42)
# Entraîner le modèle
model.fit(X_train, y_train)


# Evaluation du modèle
# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)
# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")


# Visualisation des résultats
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=['Pas de crise', 'Crise'], filled=True)
plt.show()

