# Depurar ambiente de trabajo ====
# Limpiar el workspace
rm(list = ls())
# Limpiar area de graficos
dev.off()
# Limpiar la consola
cat("\014")
# Quitar notacion cientifica
options(scipen=999)
# Numero de digitos
options(digits = 3)

# Librerias ====
library(DataExplorer)
library(VIM)
library(naniar)
library(caret)
library(cowplot)
library(missForest)
library(dummies)
library(tictoc)
library(data.table)
library(DMwR2)
library(arules)
library(funModeling)
library(ggplot2)
library(plotly)
library(dplyr)
library(RANN)
# Data ====
datos<-read.csv("Mod1/Data/loan_prediction-II.csv",
                stringsAsFactors = T, 
                sep=";",
                na.strings = "")

# Estructura de los datos
str(datos)

# Eliminando la columna del ID
datos$Loan_ID <- NULL
datos

# Declarar la variable Credit_History como factor
datos$Credit_History <- as.factor(datos$Credit_History)
levels(datos$Credit_History)  <- c("Malo","Bueno")

# Evaluando la variable target Loan_Status
table(datos$Loan_Status)

# Proporcion de los datos Loan_Status
prop.table(table(datos$Loan_Status))

str(datos)

# Datos Perdidos ====
## Exploracion de datos perdidos con DataExplorer y VIM ====
plot_missing(datos)

# Para ver cuantas filas tienen valores perdidos
# Cuantas filas tienen por lo menos 1 dato perdido
rmiss <- which(rowSums(is.na(datos))!=0,
               arr.ind=T)
length(rmiss)
# Se tienen 2590 filas con al menos 1 dato perdido

# Porcentaje de valores perdidos
length(rmiss)*100/dim(datos)[1]
# 22.51% de los datos

# Total de datos perdidos
sum(is.na(datos))   

# Graficar la cantidad de valores perdidos usando VIM
windows()
graf_perdidos1 <- aggr(datos,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

matrixplot(datos,
           main="Matrix Plot con Valores Perdidos",
           cex.axis = 0.6,
           ylab = "registro")


# Exploracion de datos perdidos con naniar
# Total de datos perdidos
n_miss(datos)      
prop_miss(datos)   # 2831/(11500*13)

# Datos perdidos por variable
miss_var_summary(datos)
n_miss(datos$Gender)
n_miss(datos$Loan_Status)

# Visualizando los datos perdidos por casos y variables
vis_miss(datos)

vis_miss(datos, cluster = TRUE)

# Variables categoricas consideracion con Missing values
table(datos$Gender)
addmargins(table(datos$Gender))
# Se obtiene 11264 datos de los 11500

# Para lecturar con los datos NA inclusive
table(datos$Gender, useNA = "always")
addmargins(table(datos$Gender,useNA = "always"))
# Se tienen 236 datos NA

# Variables Numericas consideraciones con missing data
sum(datos$LoanAmount)
# Al tener datos perdidos nos arroja NA como resultado
# Se debe de omitir los valores NA
sum(datos$LoanAmount,na.rm=T)

# Preprocesamiento de los datos ====
# Opcion 1
# Imputando los valores perdidos cuantitativos usando k-nn
# y estandarizando las variables numericas (por defecto)

# Empleando caret
preProcValues1 <- preProcess(datos,
                             method=c("knnImpute"))

# Asumiendo que no hay datos perdidos
# preProcValues1 <- preProcess(datos,
#                              method=c("center", "scale"))
#                              method=c("range"))
# Otras opciones: range , bagImpute, medianImpute, pca
#                 k= 3

preProcValues1

datos_transformado1 <- predict(preProcValues1, datos)

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

# Graficar la cantidad de valores perdidos
graf_perdidos2 <- aggr(datos_transformado1,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)


# Imputar de valores categoricos usando el algoritmo Random Forest
# Usando libreria missForest
# Este algoritmo puede imputar datos numericos tambien
set.seed(123)
impu_cate<- missForest(datos_transformado1)
datos_transformado1 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

plot_missing(datos_transformado1)

# Identificando variables con variancia cero o casi cero
# Del paquete caret
nearZeroVar(datos_transformado1, saveMetrics= TRUE)
# zeroVar: a vector of logicals for whether the predictor has only one distinct value
# nzv: a vector of logicals for whether the predictor is a near zero variance predictor


# freqRatio              percent     Unique zeroVar   nzv
# Gender                 7.85        0.0174   FALSE FALSE
# Married                3.56        0.0174   FALSE FALSE
# Dependents             2.96        0.0348   FALSE FALSE
# Education              3.20        0.0174   FALSE FALSE
# Self_Employed          8.51        0.0174   FALSE FALSE
# ApplicantIncome        1.02        4.3913   FALSE FALSE
# CoapplicantIncome      3.04        3.4000   FALSE FALSE
# LoanAmount             1.09        1.8435   FALSE FALSE
# Loan_Amount_Term      11.68        0.0957   FALSE FALSE
# Credit_History         5.66        0.0174   FALSE FALSE
# Property_Area          1.18        0.0261   FALSE FALSE
# Nacionality          337.24        0.0174   FALSE  TRUE
# Loan_Status            2.51        0.0174   FALSE FALSE

table(datos_transformado1$Nacionality)
datos_transformado1$Nacionality <- NULL

# Verificando freqRatio y percentUnique para Gender ====
table(datos_transformado1$Gender)

# Female   Male 
#   1299  10201 

# freqRatio     = (10201/1299)   = 7.85
# percentUnique = (2/11500)*100  = 0.01739130  

# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)

table(datos_transformado1$Property_Area)


# Creando variables dummies ====
# Usando el paquete dummies
datos_transformado1 <- dummy.data.frame(datos_transformado1,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))

# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)
# Al tener Variables en exceso por transformarla a dummy, se eliminan las variables extras
# GenreFemale y GenreMale - Solo 1 de estas aplica para el modelo predictor.
datos_transformado1 <- datos_transformado1[, -c(2,4,8,10,12,18,21)]

str(datos_transformado1)

# Opcion 2
# Imputando los valores perdidos cuantitativos usando k-nn, 
# y aplicando transformacion Box-Cox a las variables numericas
set.seed(123)
preProcValues2 <- preProcess(datos, method=c("knnImpute","BoxCox"))

preProcValues2

datos_transformado2 <- predict(preProcValues2, datos)

# Distribucion de las variables numericas sin transformar
ggplot(datos,
       aes(ApplicantIncome)) + 
  geom_histogram(color="black",
                 fill="white") -> g1 ; g1

ggplot(datos,
       aes(CoapplicantIncome)) + 
  geom_histogram(color="black",
                 fill="white") -> g2 ; g2

ggplot(datos,
       aes(LoanAmount)) + 
  geom_histogram(color="black",
                 fill="white") -> g3 ; g3

ggplot(datos,
       aes(Loan_Amount_Term)) + 
  geom_histogram(color="black",
                 fill="white") -> g4 ; g4

# Distribucion de las variables numericas transformadas
ggplot(datos_transformado2, 
       aes(ApplicantIncome)) + 
  geom_histogram(color="black",
                 fill="darksalmon") -> g5 ; g5

ggplot(datos_transformado2, 
       aes(CoapplicantIncome)) + 
  geom_histogram(color="black",
                 fill="darksalmon") -> g6 ; g6

ggplot(datos_transformado2,
       aes(LoanAmount)) + 
  geom_histogram(color="black",
                 fill="darksalmon") -> g7 ; g7

ggplot(datos_transformado2, 
       aes(Loan_Amount_Term)) + 
  geom_histogram(color="black",
                 fill="darksalmon") -> g8 ; g8

# Usando cowplot
plot_grid(g1,g5,g2,g6,g3,g7,g4,g8, ncol = 2)

plot_missing(datos_transformado2)

# Imputacion de datos categoricos
# Imputar valores missing usando el algoritmo Random Forest
set.seed(123)
impu_cate <- missForest(datos_transformado2)
datos_transformado2 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado2))

plot_missing(datos_transformado2)

# Identificando variables con variancia cero o casi cero
nearZeroVar(datos_transformado2, saveMetrics= TRUE)

table(datos_transformado2$Nacionality)
datos_transformado2$Nacionality <- NULL

# Verificando la estructura del archivo pre-procesado
str(datos_transformado2)


# Creando variables dummies
datos_transformado2 <- dummy.data.frame(datos_transformado2,
                                        names=c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"))

# Verificando la estructura del archivo pre-procesado
str(datos_transformado2)

datos_transformado2 <- datos_transformado2[, -c(2,4,8,10,12,18,21)]

str(datos_transformado2)


# Identificando predictores correlacionados
descrCor1 <- cor(datos_transformado2[,c(8,9,10,11)])
descrCor1

summary(descrCor1[upper.tri(descrCor1)])

# Del paquete caret
# Encontrar las variables que se encuentren por encima de 0.4
altaCorr <- findCorrelation(descrCor1, cutoff = 0.40, names=TRUE)
altaCorr
# "LoanAmount" tiene una correlacion mayor que 0.40.

# Retirando la variable 10 : LoanAmount
descrCor2 <- cor(datos_transformado2[,c(8,9,11)])
summary(descrCor2[upper.tri(descrCor2)])

datos_transformado2 <- datos_transformado2[,-10]
altaCorr2 <- findCorrelation(descrCor2, cutoff = 0.40, names=TRUE)
altaCorr2