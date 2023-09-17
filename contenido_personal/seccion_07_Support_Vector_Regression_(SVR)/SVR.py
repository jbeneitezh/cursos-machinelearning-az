# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:55:04 2023

@author: jbene
"""

# SVR
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler;
sc_X = StandardScaler();
sc_Y = StandardScaler();
x = sc_X.fit_transform(x);
y = sc_Y.fit_transform(y.reshape(-1,1));

# Ajustar la regresión con el dataset
from sklearn.svm import SVR;
regression = SVR(kernel = "rbf");
regression.fit(x, y);

# Predicción de nuestros modelos con SVR
y_pred = regression.predict(sc_X.transform([[6.5]]));
y_pred = sc_Y.inverse_transform(y_pred.reshape(-1,1));

# Visualización de los resultados del SVR
x_grid = np.arange(min(sc_X.inverse_transform(x)), max(sc_X.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1);
x_grid_t = sc_X.fit_transform(x_grid);

y_inv  = sc_Y.inverse_transform(regression.predict(x_grid_t).reshape(-1,1));

plt.scatter(sc_X.inverse_transform(x), sc_Y.inverse_transform(y), color = "red")
plt.plot(x_grid,
         y_inv, 
         color = "blue");
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
