
#Plantilla de pre-procesado
#Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importar el data set
dataset = pd.read_csv('Data.csv')


x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

#Tratamiento de los NAs
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

z=x

#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
labelencoder_X = LabelEncoder();
z[:, 0] = labelencoder_X.fit_transform(x[:, 0]);

#Forma vieja (ya no funciona)
#onehotencoder = OneHotEncoder(categorical_features = [0]);
#x = onehotencoder.fit_transform(x).toarray();

from sklearn.compose import ColumnTransformer;
ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
        remainder='passthrough'
    );
x = np.array(ct.fit_transform(x), dtype=np.float32);

labelencoder_Y = LabelEncoder();
y = labelencoder_Y.fit_transform(y);

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split;
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                        random_state=0);

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
x_train = sc_X.fit_transform(x_train);
x_test  = sc_X.transform(x_test);
