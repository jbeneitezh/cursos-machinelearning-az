# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:41:51 2023

@author: jbene
"""

#Regresión Lineal Simple
#Plantilla de pre-procesado
#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('Salary_Data.csv')


x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values



#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split;
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, 
                                        random_state=0);

#Crear modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression;
regression = LinearRegression();
regression.fit(x_train, y_train);


#Predecir el conjunto de test
y_pred = regression.predict(x_test);


#Visualizar los resultados de entrenamiento
plt.scatter(x_train, y_train, color = "red");
plt.plot(x_train, regression.predict(x_train), color = "blue");
plt.title("Sueldo vs Años Experiencia (Conjunto de entrenamiento");
plt.xlabel("Años de experiencia");
plt.ylabel("Sueldo ($)");
plt.show();

#Visualizar los resultados de test
plt.scatter(x_test, y_test, color = "green");
plt.plot(x_train, regression.predict(x_train), color = "blue");
plt.title("Sueldo vs Años Experiencia (Conjunto de Testing");
plt.xlabel("Años de experiencia");
plt.ylabel("Sueldo ($)");
plt.show();
