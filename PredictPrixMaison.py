# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Section 1 : Chargement des données
def load_data():
    """Charge le fichier CSV contenant les données du marché immobilier américain"""
    USAhousing = pd.read_csv('USA_Housing.csv')
    return USAhousing

# Section 2 : Aperçu des données
def show_data_info(USAhousing):
    """Affiche les 5 premières lignes et des informations générales sur le dataset"""
    print(USAhousing.head(), "\n")
    print(USAhousing.info())

# Section 3 : Visualisation avec Seaborn
def visualize_data(USAhousing):
    """Visualise les relations entre les différentes variables du dataset avec pairplot"""
    sns.pairplot(USAhousing)
    plt.show()

# Section 4 : Préparer les données
def prepare_data(USAhousing):
    """
    Prépare les données pour l'entraînement du modèle de régression linéaire.
    Sépare les features indépendantes (X) et la variable cible (y), puis effectue
    un découpage en ensembles d'entraînement et de test.
    Entraîne un modèle de régression linéaire sur les données d'entraînement et affiche
    les coefficients du modèle.
    """
    # Sélection des features explicatives X et de la cible youi
    X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                    'Avg. Area Number of Bedrooms', 'Area Population']]
    y = USAhousing['Price']

    # Division du dataset en ensembles d'entraînement (70%) et de test (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Instanciation du modèle de régression linéaire
    regressor = LinearRegression()

    # Entraînement du modèle sur les données d'entraînement
    regressor.fit(X_train, y_train)

    # Affichage de l'ordonnée à l'origine (intercept) et des coefficients du modèle
    print("Intercept (b0) : ", regressor.intercept_)
    print("Coefficients des features : ", regressor.coef_)

    # Conversion des coefficients en DataFrame pour une meilleure lisibilité
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    print("\nCoefficients des features dans le modèle :\n", coeff_df)

    # Retourne le modèle entraîné ainsi que les données pour l'évaluation
    return regressor, X_test, y_test

# Section 5 : Faire des prédictions et évaluer le modèle
def prediction(regressor, X_test, y_test):
    """
    Utilise le modèle de régression linéaire pour prédire les prix sur l'ensemble de test.
    Affiche un graphique de comparaison entre les prix réels et les prix prédits, ainsi que
    les mesures d'évaluation du modèle (MAE, MSE, RMSE, R²).
    """
    # Prédiction des prix sur l'ensemble de test
    y_predict = regressor.predict(X_test)

    # Visualisation de la comparaison entre les prix réels et prédits
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_predict, edgecolor='k', alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title("Comparaison entre les prix réels et prédits", fontsize=16)
    plt.xlabel("Prix réels", fontsize=14)
    plt.ylabel("Prix prédits", fontsize=14)
    plt.show()

    # Calcul des métriques d'évaluation
    print('\nMean Absolute Error (MAE) : ', metrics.mean_absolute_error(y_test, y_predict))
    print('Mean Squared Error (MSE) : ', metrics.mean_squared_error(y_test, y_predict))
    print('Root Mean Squared Error (RMSE) : ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
    print('R² Score : ', metrics.r2_score(y_test, y_predict))

if __name__ == "__main__":
    # Chargement et préparation des données
    data = load_data()
    show_data_info(data)
    #visualize_data(data)

    # Entraînement du modèle et évaluation
    regressor, X_test, y_test = prepare_data(data)
    prediction(regressor, X_test, y_test)
