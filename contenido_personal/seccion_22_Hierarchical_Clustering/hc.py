# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:05:48 2023

@author: jbene
"""

#Clustering Jerárquico
import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd

#Importar los datos del centro comercial con pandas
dataset = pd.read_csv("Mail_Customers.csv");
x = dataset.iloc[:, [3, 4]].values;

#Utilizar el dendograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch;
dendogram = sch.dendrogram(sch.linkage(x, method = "ward"));
plt.title("Dendograma");
plt.xlabel("Clientes");
plt.ylabel("Distancia Euclídea")
plt.show()

#Ajustar el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward");
y_hc = hc.fit_predict(x);

#Visualización de los clústers
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = "red", label = "Cluster1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = "green", label = "Cluster3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster5")
plt.title("Cluster de clientes");
plt.xlabel("Ingresos anuales (en miles de $)");
plt.ylabel("Puntuación de Gastos (1-100");
plt.legend()
plt.show()

