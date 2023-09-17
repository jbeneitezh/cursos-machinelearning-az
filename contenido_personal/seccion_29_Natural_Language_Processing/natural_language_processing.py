# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 03:43:55 2023

@author: jbene
"""

#Natural Language Processing

#importar librerías
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3);

#Limpieza de texto
import re;
import nltk;
nltk.download('stopwords');
from nltk.corpus import stopwords;
from nltk.stem.porter import PorterStemmer;

reviews     = dataset.iloc[:, 0].values;
n = dataset.shape[0];
corpus = [];
for i in range(0, n):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]); #Deja sólo los caracteres que hemos pedido
  review = review.lower();
  review = review.split(); #Separa la frase en un array de palabra
  ps = PorterStemmer(); #Este transforma palabras. Por ejemplo de loved a love
  #Quitamos las palabras 'irrelevantes' en inglés
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]; 
  review = ' '.join(review); #transforma la lista en un string
  corpus.append(review);


#Crear el Bag of words
from sklearn.feature_extraction.text import CountVectorizer;
cv = CountVectorizer(max_features = 1500);
X = cv.fit_transform(corpus).toarray();
y = dataset.iloc[:, 1].values;


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC;
classifier = SVC(kernel = "rbf", random_state = 0);
classifier.fit(X_train, y_train);


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test);

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Kernel (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Kernel (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

