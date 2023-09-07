setwd("D:/vs_wks/cursos-machinelearning-az/contenido_personal/seccion_2/parte1_preprocesado")
#Plantilla de pre-procesado
#Importar el dataset
dataset = read.csv('Data.csv')


#Tratamiento de los valores NA
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#Codificar las variables categóricas
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0, 1))

#Dividir los datos en conjunto de entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Escalado de valores
train_set[, 2:3] = scale(train_set[, 2:3])
test_set [, 2:3] = scale(test_set [, 2:3])

