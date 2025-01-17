setwd("D:/vs_wks/cursos-machinelearning-az/contenido_personal/seccion_4_regresi�n_lineal_simple")

#Regresi�n lineal simple
#Plantilla de pre-procesado
#Importar el dataset
dataset = read.csv('Salary_Data.csv')



#Dividir los datos en conjunto de entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Ajustar el modelo de regresi�n lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience, data = train_set)

#Predecir resultados con el conjunto de test
y_pred = predict(regressor, newdata = test_set)
 

#Visualizaci�n de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary),
             colour = "red") +
  geom_line(aes(x = train_set$YearsExperience, y = predict(regressor, newdata = train_set)),
            colour = "blue") +
  ggtitle("Suelo vs Experiencia (Conjunto de entrenamiento)")+
  xlab("A�os de experiencia") +
  ylab("Sueldo ($)")

#Visualizaci�n de los resultados del conjunto de testing.
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = "red") +
  geom_line(aes(x = train_set$YearsExperience, y = predict(regressor, newdata = train_set)),
            colour = "blue") +
  ggtitle("Suelo vs Experiencia (Conjunto de testing)")+
  xlab("A�os de experiencia") +
  ylab("Sueldo ($)")


