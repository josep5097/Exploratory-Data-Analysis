

# Limpiar el workspace
rm(list = ls())
dev.off()

# Paquetes
library(caret)
library(caTools)
library(ggplot2)
library(MLmetrics)
library(dplyr)
library(pROC)

# Prediccion de fuga de clientes ====
# Variables predictoras:
# Tasa: Tasa de interes de la cuenta CTS
# Saldo_soles: Monto de Saldo de la cuenta CTS, en Soles.
# Edad: Edad del cliente en anios
# EstadoCivil: Div.Sol.Viu = Divorciado, 
#              Soltero y Viudo y Cas.Conv = Casado, Conviviente
# Region: NORTE.SUR, ORIENTE, CENTRO, LIMA_CALLAO
# CrossSell: Numero de productos vigentes con el banco,tanto pasivos o activos
# Ratio.Ant: Ratio Ant_Cts / Ant_Banco  
#     Ant_Banco: Tiempo de antig?edad del cliente (en meses)
#     Ant_Cts: Tiempo de antig?edad de la cuenta CTS (en meses)
#
# Variable dependiente:
# 0 = cliente no fugado
# 1 = cliente fugado

# Data ====
datos.cts <- read.table("Mod5/Data/fuga_desbalanceada_cts.csv",
                        sep=";",
                        dec=",",
                        header=T,stringsAsFactors = T)

str(datos.cts)

# Quitar el campo ID
datos.cts$Id <- NULL

# Manipulacion
datos.cts$Fuga <- as.factor(datos.cts$Fuga)
levels(datos.cts$Fuga) <- c("No_Fuga","Si_Fuga")

str(datos.cts)

round(prop.table(table(datos.cts$Fuga))*100,2)

contrasts(datos.cts$Fuga)


# Separar bases ====
set.seed(123) 
index   <- createDataPartition(datos.cts$Fuga, 
                               p=0.8, list=FALSE)

imbal_train    <- datos.cts[ index, ]
imbal_testing  <- datos.cts[-index, ]

# Verificando que se mantenga la proporcion original
addmargins(table(datos.cts$Fuga))
round(prop.table(table(datos.cts$Fuga))*100,2)

addmargins(table(imbal_train$Fuga))
round(prop.table(table(imbal_train$Fuga))*100,2)

addmargins(table(imbal_testing$Fuga))
round(prop.table(table(imbal_testing$Fuga))*100,2)

# Generando nueva data
# Undersampling
set.seed(123)
under_train <- downSample(x = imbal_train[,c(1:7)], 
                          y = imbal_train$Fuga,
                          yname="Fuga")

addmargins(table(under_train$Fuga))

# OverSampling
set.seed(123)
over_train <- upSample(x = imbal_train[, c(1:7)],
                       y = imbal_train$Fuga,
                       yname="Fuga")

addmargins(table(over_train$Fuga))

# Modelo con validacion cruzada con v= 10
#    Usar como indicador el Accuracy

# Relacion de modelos 
# Nombres de modelos que ofrece la libreria Caret
names(getModelInfo())

# Relacion de parametros ajustar de un modelo
modelLookup(model="rpart")
modelLookup(model="xgbTree")
modelLookup(model="glm")

# Metodo de validacion para todos los modelos  
ctrl <- trainControl(method="cv",
                     number=10)


# En todos los modelos train(), usar relevel(Fuga,ref="Si_Fuga") ~ .
# en vez de Fuga ~ .

# Datos Desbalanceados
set.seed(123)
modelo_orig   <- train(Fuga ~ ., 
                       data = imbal_train, 
                       method="glm",
                       family="binomial", 
                       trControl = ctrl,
                       metric="Accuracy")

modelo_orig

summary(modelo_orig)
varImp(modelo_orig)
plot(varImp(modelo_orig))
# Modelo con datos balanceados (undersampling)
set.seed(123)
modelo_under  <- train(Fuga ~ ., 
                       data = under_train, 
                       method="glm", 
                       family="binomial", 
                       trControl = ctrl, 
                       metric="Accuracy")
modelo_under

# Modelo con los datos balanceados (oversampling)
set.seed(123)
modelo_over    <- train(Fuga ~ ., 
                        data = over_train, 
                        method="glm", 
                        family="binomial", 
                        trControl = ctrl, 
                        metric="Accuracy")
modelo_over


# Comparando los modelos
modelos  <- list(original = modelo_orig,
                 under    = modelo_under,
                 over     = modelo_over)

comparacion_modelos <- resamples(modelos)
summary(comparacion_modelos)

dotplot(comparacion_modelos)

bwplot(comparacion_modelos)

# Predicciones
# Prediccion del modelo_orig en la data testing
# Prob con el punto de corte 0.5
# Entrega dos respuestas, la prob del positivo y 
# del negativo
proba.modelo_orig <- predict(modelo_orig,
                             newdata = imbal_testing, 
                             type="prob")
# Nos quedamos con la probabilidad del exito de fuga
proba.modelo_orig <- proba.modelo_orig[,2]
# Sin el type prob, nos predice la clase
# es decir la variable categorica
clase.modelo_orig <- predict(modelo_orig,
                             newdata = imbal_testing )

# Matriz de confusion mediante caret
result1 <- caret::confusionMatrix(clase.modelo_orig,
                                  imbal_testing$Fuga,
                                  positive="Si_Fuga")

result1
result1$byClass["Sensitivity"] 
result1$byClass["Specificity"] 
result1$overall["Accuracy"]
# No es un buen modelo debido a que se posee una
# sensibilidad de cero

# Area bajo la curva
# library(caTools)
colAUC(proba.modelo_orig,imbal_testing$Fuga,plotROC = TRUE) -> auc1
abline(0, 1,col="red")
auc1

# Prediccion del modelo_under en la data testing
proba.modelo_under <- predict(modelo_under,newdata = imbal_testing, type="prob")
proba.modelo_under <- proba.modelo_under[,2]
clase.modelo_under <- predict(modelo_under,newdata = imbal_testing )

result2 <- caret::confusionMatrix(clase.modelo_under,
                                  imbal_testing$Fuga,
                                  positive="Si_Fuga")

result2
result2$byClass["Sensitivity"] 
result2$byClass["Specificity"] 
result2$overall["Accuracy"]

# Area bajo la curva
colAUC(proba.modelo_under,imbal_testing$Fuga,plotROC = TRUE) -> auc2
abline(0, 1,col="red")
auc2

# Prediccion del modelo_over en la data testing
clase.modelo_over  <- predict(modelo_over,newdata = imbal_testing )
proba.modelo_over  <- predict(modelo_over,newdata = imbal_testing, type="prob")
proba.modelo_over  <- proba.modelo_over[,2]

result3 <- caret::confusionMatrix(clase.modelo_over,
                                  imbal_testing$Fuga,
                                  positive="Si_Fuga")

result3
result3$byClass["Sensitivity"] 
result3$byClass["Specificity"] 
result3$overall["Accuracy"]

# Area bajo la curva
colAUC(proba.modelo_over,imbal_testing$Fuga,plotROC = TRUE) -> auc3
abline(0, 1,col="red")
auc3

# Resumiendo los 3 modelos
balanceo <- c("Sin Balancear","Undersampling",
                   "Oversampling")

sensibilidad  <- c(result1$byClass["Sensitivity"],
                   result2$byClass["Sensitivity"],
                   result3$byClass["Sensitivity"])

especificidad <- c(result1$byClass["Specificity"],
                   result2$byClass["Specificity"],
                   result3$byClass["Specificity"])

accuracy      <- c(result1$overall["Accuracy"],
                   result2$overall["Accuracy"],
                   result3$overall["Accuracy"])

auc           <- c(auc1,auc2,auc3)

comparacion <- data.frame(balanceo,
                          sensibilidad,
                          especificidad,
                          accuracy,auc)

comparacion
