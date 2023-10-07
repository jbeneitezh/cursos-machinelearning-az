# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 01:01:52 2023

@author: jbene
"""

#Parte I - Construir el modelo de CNN

#Importar las librerías y paquetes
from keras.models import Sequential;
from keras.layers import Conv2D;
from keras.layers import MaxPooling2D;
from keras.layers import Flatten;
from keras.layers import Dense;

#Inicializar la CNN
classifier = Sequential();

#Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, 
                      kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), 
                      activation = "relu")
               );

#Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)));

#Paso 3 - Flattening
classifier.add(Flatten());

#Paso 4 - Full Connection
classifier.add(Dense(units = 128, activation = "relu"));
classifier.add(Dense(units = 1  , activation = "sigmoid"));

#Compilar la CNN
classifier.compile(optimizer = "adam", 
                   loss = "binary_crossentropy", 
                   metrics = ["accuracy"]
                   );

#Parte 2 - Ajustar la CNN a las imágenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=40,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=8,
                                                class_mode='binary')

classifier.fit_generator(training_dataset,
                        steps_per_epoch=200,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=1000)