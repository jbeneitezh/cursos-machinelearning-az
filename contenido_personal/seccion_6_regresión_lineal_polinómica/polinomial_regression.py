# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:13:27 2023

@author: jbene
"""

#Regresión polinómica




#Plantilla de pre-procesado
#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')


x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression;
lin_reg = LinearRegression();
lin_reg.fit(x, y);

#Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures;
poly_reg = PolynomialFeatures(degree = 4);
x_poly = poly_reg.fit_transform(x);

lin_reg2 = LinearRegression();
lin_reg2.fit(x_poly, y);

#Visualización de los resultados del Modelo Lineal
x_grid = np.arange(min(x), max(x), 0.1);
x_grid = x_grid.reshape(len(x_grid), 1);
plt.scatter(x, y, color = "red");
plt.plot(x, lin_reg.predict(x), color = "blue");
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = "green");
plt.title("Modelo de Regresión Lineal");
plt.xlabel("Posición del empleado");
plt.ylabel("Sueldo (en $)");
plt.show();


#Visualización de los resultados del Modelo Polinómico
plt.scatter(x, y, color = "red");
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = "blue");
plt.title("Modelo de Regresión Polinómica");
plt.xlabel("Posición del empleado");
plt.ylabel("Sueldo (en $)");
plt.show();

#Predicción de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
