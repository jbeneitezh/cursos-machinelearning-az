# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 21:45:17 2023

@author: jbene
"""

# Upper Confidence Bound (UCB)


#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#Algoritmo de Upper Confidence Bound (UCB)
import math;
N = 10000; #usuarios del dataset
d = 10;    #número de columnas
number_of_selections = [0] * d;
sum_of_rewards = [0] * d;
ads_selected = [];
total_reward = 0;
for n in range(0, N):
    max_upper_bound = 0;
    ad = 0;
    for i in range(0, d):
        if(number_of_selections[i]>0):
            average_reward = sum_of_rewards[i] / number_of_selections[i];
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i]);
            upper_bound = average_reward + delta_i;
        else:
            upper_bound = 1e400;
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound;
            ad = i;
    ads_selected.append(ad);
    number_of_selections[ad]=number_of_selections[ad]+1;
    reward = dataset.values[n, ad];
    sum_of_rewards[ad]=sum_of_rewards[ad]+reward;
    total_reward = total_reward + reward;
    
#Histograma de resultados
plt.hist(ads_selected);
plt.title("Histograma de anuncios");
plt.xlabel("ID del Anuncio");
plt.ylabel("Frecuencia de visualización del anuncio");
plt.show();


