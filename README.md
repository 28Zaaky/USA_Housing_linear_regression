# Prédiction des Prix Immobiliers aux États-Unis  

Ce projet utilise un modèle de **régression linéaire multiple** pour prédire les prix de l'immobilier en fonction de plusieurs variables explicatives, comme le revenu moyen d'une zone, l'âge moyen des maisons ou la population d'une région.  

## Fonctionnalités  
- **Chargement et exploration des données** : Analyse des données avec affichage des informations générales et visualisation des relations entre les variables.  
- **Visualisation** : Utilisation de `Seaborn` pour examiner les corrélations entre les différentes caractéristiques et la cible.  
- **Régression linéaire multiple** : Entraînement d'un modèle sur des données d'entraînement (70 %) et évaluation sur des données de test (30 %).  
- **Évaluation des performances** : Calcul de métriques comme MAE, MSE, RMSE et R² pour évaluer la précision du modèle.  
- **Visualisation des prédictions** : Comparaison graphique entre les prix réels et les prix prédits.  

## Données utilisées  
Le dataset utilisé est **`USA_Housing.csv`**, qui contient les colonnes suivantes :  
- `Avg. Area Income` : Revenu moyen des résidents d'une région.  
- `Avg. Area House Age` : Âge moyen des maisons dans la région.  
- `Avg. Area Number of Rooms` : Nombre moyen de pièces dans les maisons.  
- `Avg. Area Number of Bedrooms` : Nombre moyen de chambres dans les maisons.  
- `Area Population` : Population totale de la région.  
- `Price` : Prix des maisons (variable cible).  

## Prérequis  
- **Python 3**  
- Bibliothèques nécessaires :  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `scikit-learn`  

## Métriques d'évaluation
-**Le modèle est évalué à l'aide des métriques suivantes :**

- `MAE (Mean Absolute Error) : Erreur moyenne absolue.`
- `MSE (Mean Squared Error) : Erreur quadratique moyenne.`
- `RMSE (Root Mean Squared Error) : Racine carrée de l'erreur quadratique moyenne.`
- `R² (Coefficient de détermination) : Proportion de la variance expliquée par le modèle.`
