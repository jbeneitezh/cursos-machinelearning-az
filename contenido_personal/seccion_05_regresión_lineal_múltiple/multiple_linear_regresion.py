# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:46:00 2023

@author: jbene
"""

#Regresión Lineal Múltiple

#Plantilla de pre-procesado
#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('50_Startups.csv')


x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values



#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
labelencoder_X = LabelEncoder();
x[:, 3] = labelencoder_X.fit_transform(x[:, 3]);

#Forma vieja (ya no funciona)
#onehotencoder = OneHotEncoder(categorical_features = [0]);
#x = onehotencoder.fit_transform(x).toarray();

from sklearn.compose import ColumnTransformer;
ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
        remainder='passthrough'
    );
x = np.array(ct.fit_transform(x), dtype=np.float32);

#Evitar la trampa de las variables ficticias (eliminando una de las variables dummy)
x = x[:, 1:];

#labelencoder_Y = LabelEncoder();
#y = labelencoder_Y.fit_transform(y);

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split;
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                        random_state=0);

#Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression;
regression = LinearRegression();
regression.fit(x_train, y_train);


#Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(x_test);


#Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm;
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1); #añade columna 1 al principio

sl = 0.05; #nivel de significación

x_opt = x[:, [0, 1, 2, 3, 4, 5]];
regression_ols = sm.OLS(endog = y, exog = x_opt).fit();
regression_ols.summary()

#Quitamos variables hasta que todas tengan un p-valor por debajo de 0.05
x_opt = x[:, [0, 1, 3, 4, 5]];
regression_ols = sm.OLS(endog = y, exog = x_opt).fit();
regression_ols.summary()


x_opt = x[:, [0, 3, 4, 5]];
regression_ols = sm.OLS(endog = y, exog = x_opt).fit();
regression_ols.summary()


x_opt = x[:, [0, 3, 5]];
regression_ols = sm.OLS(endog = y, exog = x_opt).fit();
regression_ols.summary()

x_opt = x[:, [0, 3]];
regression_ols = sm.OLS(endog = y, exog = x_opt).fit();
regression_ols.summary()



#Automatización de la eliminación hacia atrás
sl = 0.05; #nivel de significación
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
x_opt2 = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt2, sl)
x_Modeled
