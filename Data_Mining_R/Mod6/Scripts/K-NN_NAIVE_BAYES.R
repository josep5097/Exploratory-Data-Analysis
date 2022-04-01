# Emplear un modelo de prediccion con el algoritmo 
# de knn y naive bayes

# Limpiear workspace
rm(list = ls())

# Limpiar area de graficos
dev.off()

# Limpiar la consola
cat("\014")
options(scipen=999)
options(digits = 4)

# Librerias ====
library(foreign)
library(dummies)
library(caret)
library(gmodels)
library(caTools)
library(ggplot2)

# Descripcion
# Variables Independentes:
#       EDAD    (Edad del cliente en anios)  
#       SEXO    (Sexo del cliente:
#               1=Fememino 2=Masculino)
#       CIVIL   (Estado civil del cliente, 
#               1=Casado 2=Soltero) 
#       HIJOS   (Numero de hijos del cliente)
#       INGRESO (Ingresos anuales del cliente)
#       AUTO    (Si el cliente es dueno de un auto, 
#                1=Si 2=No)  
#
# Variable de identificacion: 
#           ID      (Codigo del cliente)


# Data ====
datos <-read.spss("Mod6/Data/Churn-arboles.sav",
                  use.value.labels=TRUE, 
                  to.data.frame=TRUE)

attr(datos,"variable.labels")<- NULL

str(datos)

# Quitar variable ID
datos$ID <- NULL
str(datos)

# Levels
levels(datos$SEXO)
levels(datos$SEXO)  <- c("Fem","Masc")
levels(datos$CIVIL) <- c("Casado","Soltero")
levels(datos$AUTO)  <- c("Si","No")
levels(datos$CHURN) <- c("Actual","Fuga")

# Variables Dummies
datos <- dummy.data.frame(datos,names=c("SEXO","CIVIL","AUTO"))
str(datos)
datos <- datos[,-c(3,5,9)]

# Dividir la informacion
str(datos)                              
set.seed(123) 
index <- createDataPartition(datos$CHURN, 
                             p=0.7, 
                             list=FALSE)
training <- datos[ index, ]
testing  <- datos[-index, ]
prop.table(table(datos$CHURN))
prop.table(table(training$CHURN))
prop.table(table(testing$CHURN))


# Algoritmo k-nn con Validacion Cruzada 
# Relacion de parametros a ajustar de un modelo
modelLookup(model='knn')

ctrl <- trainControl(method="cv", 
                     number=10)
set.seed(123)
modelo_knn <- train(CHURN ~ ., 
                    data = training, 
                    method = "knn", #kknn
                    preProcess=c("range"), # Estandarizar los datos entre 0 y 1
                    trControl = ctrl, #crossvalidation 
                    tuneLength = 5, # por default
#                   tuneGrid = expand.grid(k=seq(1,40,1)),
                    metric="Accuracy") # k con mayor accuracy

modelo_knn

plot(modelo_knn)
varImp(modelo_knn)
plot(varImp(modelo_knn))

PROBA.KNN <- predict(modelo_knn,
                     newdata = testing, 
                     type="prob")
head(PROBA.KNN)
PROBA.KNN <- PROBA.KNN[,2]

CLASE.KNN <- predict(modelo_knn,newdata = testing )
head(CLASE.KNN)

# Evaluando la performance del modelo k-nn
# Tabla de clasificacion
# Empleando library(gmodels)
CrossTable(x = testing$CHURN,
           y = CLASE.KNN,
           prop.t=FALSE, 
           prop.c=FALSE, 
           prop.chisq = FALSE)

addmargins(table(Real=testing$CHURN,Clase_Predicha=CLASE.KNN))
prop.table(table(Real=testing$CHURN,Clase_Predicha=CLASE.KNN),1)

# Calcular el accuracy
accuracy_knn <- mean(testing$CHURN==CLASE.KNN) ; accuracy_knn

# Calcular el error de mala clasificaci?n
error <- mean(testing$CHURN!=CLASE.KNN) ; error

# Tabla de clasificacion usando paquete caret
cm_knn <- caret::confusionMatrix(CLASE.KNN,
                                 testing$CHURN,
                                 positive="Fuga")
cm_knn

cm_knn$table

cm_knn$byClass["Sensitivity"] 
cm_knn$byClass["Specificity"] 
cm_knn$overall["Accuracy"]

# Curva ROC usando el paquete caTools
colAUC(PROBA.KNN,testing$CHURN,plotROC = TRUE) -> auc_knn
abline(0,1,col="red")
auc_knn

# Tuneando el modelo
set.seed(123)
modelo_knn <- train(CHURN ~ ., 
                    data = training, 
                    method = "knn", #kknn
                    preProcess=c("range"), # Estandarizar los datos entre 0 y 1
                    trControl = ctrl, #crossvalidation 
                    #tuneLength = 5, # por default
                    tuneGrid = expand.grid(k=seq(1,40,1)),
                    metric="Accuracy") # k con mayor accuracy

PROBA.KNN <- predict(modelo_knn,
                     newdata = testing, 
                     type="prob")
head(PROBA.KNN)
PROBA.KNN <- PROBA.KNN[,2]

CLASE.KNN <- predict(modelo_knn,newdata = testing )
head(CLASE.KNN)

# Calcular el accuracy
accuracy_knn <- mean(testing$CHURN==CLASE.KNN) ; accuracy_knn

# Calcular el error de mala clasificaci?n
error <- mean(testing$CHURN!=CLASE.KNN) ; error

# Tabla de clasificacion usando paquete caret
cm_knn <- caret::confusionMatrix(CLASE.KNN,
                                 testing$CHURN,
                                 positive="Fuga")
cm_knn

cm_knn$table

cm_knn$byClass["Sensitivity"] 
cm_knn$byClass["Specificity"] 
cm_knn$overall["Accuracy"]

# Curva ROC usando el paquete caTools
colAUC(PROBA.KNN,testing$CHURN,plotROC = TRUE) -> auc_knn
abline(0,1,col="red")
auc_knn

modelo_knn
plot(modelo_knn)



# Algoritmo Naive Bayes con Validacion Cruzada

# Parametros a ajustar de un modelo
modelLookup(model='nb')  # NaiveBayes {klaR}

set.seed(123)
ctrl <- trainControl(method="cv", 
                     number=10)

modelo_nb <- train(CHURN ~ ., 
                   data = training, 
                   method = "nb", 
                   preProcess=c("range"),
                   trControl = ctrl, 
                   tuneLength = 2,
                   metric="Accuracy" )

modelo_nb 

plot(modelo_nb)

varImp(modelo_nb)

plot(varImp(modelo_nb))

PROBA.NB <- predict(modelo_nb,
                    newdata = testing, 
                    type="prob")
head(PROBA.NB)
PROBA.NB <- PROBA.NB[,2]

CLASE.NB <- predict(modelo_nb,newdata = testing )
head(CLASE.NB)


# Evaluando la performance del modelo Naive Bayes

# Tabla de clasificacion
# library(gmodels)
CrossTable(x = testing$CHURN,
           y = CLASE.NB,
           prop.t=FALSE, 
           prop.c=FALSE, 
           prop.chisq = FALSE)

addmargins(table(Real=testing$CHURN,Clase_Predicha=CLASE.NB))
prop.table(table(Real=testing$CHURN,Clase_Predicha=CLASE.NB),1)

# Calcular el accuracy
accuracy_nb <- mean(testing$CHURN==CLASE.NB) ; accuracy_nb

# Calcular el error de mala clasificaci?n
error <- mean(testing$CHURN!=CLASE.NB) ; error

# Tabla de clasificaci?n usando paquete caret
library(caret)
cm_nb <- caret::confusionMatrix(CLASE.NB,
                                testing$CHURN,
                                positive="Fuga")
cm_nb

cm_nb$table

cm_nb$byClass["Sensitivity"] 
cm_nb$byClass["Specificity"] 
cm_nb$overall["Accuracy"]

# Curva ROC usando el paquete caTools
# library(caTools)
colAUC(PROBA.NB,testing$CHURN,plotROC = TRUE) -> auc_nb
abline(0, 1,col="red")
auc_nb


#Regresion Logistica 
# Relaci?n de parametros a ajustar de un modelo
modelLookup(model='glm')

set.seed(123)
ctrl <- trainControl(method="cv", number=10)
modelo_rl <- train(CHURN ~ ., 
                   data = training, 
                   method = "glm", family="binomial", 
                   preProcess=c("range"),
                   trControl = ctrl, 
                   tuneLength = 5,
                   metric="Accuracy" )

modelo_rl

plot(modelo_rl)

varImp(modelo_rl)
plot(varImp(modelo_rl))

CLASE.RL <- predict(modelo_rl,newdata = testing )
head(CLASE.RL)

PROBA.RL <- predict(modelo_rl,newdata = testing, type="prob")
head(PROBA.RL)
PROBA.RL <- PROBA.RL[,2]


# Evaluando la performance del modelo de Regresi?n Log?stica

# Tabla de clasificacion
# library(gmodels)
CrossTable(x = testing$CHURN, 
           y = CLASE.RL,
           prop.t=FALSE, 
           prop.c=FALSE, 
           prop.chisq = FALSE)

addmargins(table(Real=testing$CHURN,Clase_Predicha=CLASE.RL))
prop.table(table(Real=testing$CHURN,Clase_Predicha=CLASE.RL),1)

# Calcular el accuracy
accuracy_rl <- mean(testing$CHURN==CLASE.RL) ; accuracy_rl

# Calcular el error de mala clasificaci?n
error <- mean(testing$CHURN!=CLASE.RL) ; error

# Tabla de clasificaci?n usando paquete caret
library(caret)
cm_rl <- caret::confusionMatrix(CLASE.RL,
                                testing$CHURN,
                                positive="Fuga")
cm_rl

cm_rl$table

cm_rl$byClass["Sensitivity"] 
cm_rl$byClass["Specificity"] 
cm_rl$overall["Accuracy"]


# Curva ROC usando el paquete caTools
# library(caTools)
colAUC(PROBA.RL,testing$CHURN,plotROC = TRUE) -> auc_rl
abline(0, 1,col="red")
auc_rl


# Comparando el entrenamiento de los tres modelos
modelos  <- list(k_nn        = modelo_knn,
                 Naive_Bayes = modelo_nb,
                 Logistica   = modelo_rl)

comparacion_modelos <- resamples(modelos)
summary(comparacion_modelos)

dotplot(comparacion_modelos)

bwplot(comparacion_modelos)

densityplot(comparacion_modelos, 
            metric = "Accuracy",
            auto.key=TRUE)



# Resumiendo y comparando los 3 algoritmos en el test

algoritmos <- c("k-nn","Naive Bayes","Log?stica")

sensibilidad  <- c(cm_knn$byClass["Sensitivity"],cm_nb$byClass["Sensitivity"],cm_rl$byClass["Sensitivity"])
especificidad <- c(cm_knn$byClass["Specificity"],cm_nb$byClass["Specificity"],cm_rl$byClass["Specificity"]) 
accuracy      <- c(accuracy_knn,accuracy_nb,accuracy_rl)
area_roc      <- c(auc_knn,auc_nb,auc_rl)

comparacion <- data.frame(algoritmos,sensibilidad, 
                          especificidad,accuracy,area_roc)

comparacion

# Histograma con el paquete ggplot2
ggplot(datos, aes(INGRESO)) + 
  geom_histogram(color="black",fill="blue") + 
  labs(title ="Distribuci?n de los ingresos de los clientes", 
       x="Ingresos", 
       y= "Frecuencia") + theme_replace() 

ggplot(datos, aes(INGRESO)) + 
  geom_density(color="black",fill="blue",adjust=0.3) + 
  labs(title ="Distribuci?n de los ingresos de los clientes", 
       x="Ingresos") + theme_replace() 
