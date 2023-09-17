# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:58:17 2023

@author: jbene
"""

#Apriori

#Plantilla de pre-procesado
#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None);
transactions = [];
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    

#Entrenar el algoritomo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualización de los resultados
results = list(rules)

results[1]


