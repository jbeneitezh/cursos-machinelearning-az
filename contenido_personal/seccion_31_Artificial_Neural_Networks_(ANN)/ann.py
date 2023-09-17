# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 02:28:34 2023

@author: jbene
"""

#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Parte 1. Pre-procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
labelencoder_X_1 = LabelEncoder();
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]);
labelencoder_X_2 = LabelEncoder();
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]);

#Forma vieja (ya no funciona)
#onehotencoder = OneHotEncoder(categorical_features = [0]);
#x = onehotencoder.fit_transform(x).toarray();

from sklearn.compose import ColumnTransformer;
ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
        remainder='passthrough'
    );
X = np.array(ct.fit_transform(X), dtype=np.float32);
X = X[:, 1:];


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#PArte 2 - Construír la RNA
#Importar Keras y librerías
import keras
from keras.models import Sequential;
from keras.layers import Dense;

#Inicializar la RNA
classifier = Sequential();


#Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, 
                     kernel_initializer = "uniform", 
                     activation = "relu", 
                     input_dim = 11)
               );

#Añadir la segunda capa oculta
classifier.add(Dense(units = 6, 
                     kernel_initializer = "uniform", 
                     activation = "relu")
               );

#Añadir la capa de salida
classifier.add(Dense(units = 1, 
                     kernel_initializer = "uniform", 
                     activation = "sigmoid")
               );

#compilar la red neuronal artificial (RNA)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]);

#Ajustasmos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100);








# Ajustar el clasificador en el Conjunto de Entrenamiento
# Crear el modelo de clasificación aquí



# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



