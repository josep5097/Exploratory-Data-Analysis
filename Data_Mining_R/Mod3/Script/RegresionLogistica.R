# Regresion Logistica ====

rm(list = ls()) ; dev.off()
options(scipen=999)

# Librerias ====
library(funModeling)
library(MASS) 
library(pROC)
library(caret)
library(foreign)
library(gmodels)
library(vcd)
library(InformationValue)
library(caTools)
library(MLmetrics)
library(ROCit)
library(Epi)

# Preparacion de los datos ====
datos.r <-read.spss("Mod3/Data/Churn-arboles.sav",
                    use.value.labels=TRUE, 
                    to.data.frame=TRUE)

# Al exportar de SPSS se generan los labels, se procede a quitarlos
attr(datos.r,"variable.labels")<- NULL

dim(datos.r)

str(datos.r)

# No considerar la variable de identificacion ID
datos.r$ID <- NULL

str(datos.r)

# Etiquetando las opciones de las variables categoricas
levels(datos.r$SEXO)
levels(datos.r$SEXO) <- c("Fem","Masc")
levels(datos.r$CIVIL)
levels(datos.r$CIVIL) <- c("Casado","Soltero")
levels(datos.r$AUTO) <- c("Si","No")
levels(datos.r$CHURN) <- c("Actual","Fuga")

table(datos.r$CHURN)
prop.table(table(datos.r$CHURN))


# Explorando las variables predictoras
# Empleando funModelling
# Reduciendo el codigo

# Obtener el nombre de los predictores
predictores <- setdiff(names(datos.r), "CHURN")
predictores

# Realizar un grafico de barras con relleno de la variable a predecir
# Se lo realiza con todas las variables 
cross_plot(datos.r, 
           input=predictores, 
           target="CHURN",
           plot_type = "percentual") 

cross_plot(datos.r, 
           input=predictores, 
           target="CHURN",
           plot_type = "both") # quantity

# Cat: Sexo, Auto, Civil
# Sexo, mujeres son más propensas a la fuga
# Para ver la categoria de referencia 
contrasts(datos.r$AUTO)
contrasts(datos.r$CIVIL)
contrasts(datos.r$SEXO)
# Variable referencia es Fem 0
# Para hacer mas facil interpretar se cambia 
# la referencia
# Para cambiar la categoria de referencia
datos.r$SEXO = relevel(datos.r$SEXO,
                       ref="Masc")

contrasts(datos.r$SEXO)
# Hombres 0
# Mujeres 1

# Para civil y auto no se realizar modificacion
# debido a que no tienen mayor influencia

contrasts(datos.r$CHURN)
# Siempre se pone 0 al valor normal
# El valor 1, es el estado de cambio

# Split de poblacion ====
# Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
# Empleando caret createDataPartition
set.seed(123) 
index    <- createDataPartition(datos.r$CHURN, 
                                p=0.7, 
                                list=FALSE)
training <- datos.r[ index, ]
testing  <- datos.r[-index, ]

# Verificando la estructura de los datos particionados
prop.table(table(datos.r$CHURN))
prop.table(table(training$CHURN))
prop.table(table(testing$CHURN))

# Modelo log?stico con todas las variables
modelo_churn <- glm(CHURN ~ . , 
                    family=binomial,
                    data=training)

summary(modelo_churn)
coef(modelo_churn)
# No se interpretan los coeficientes
# sino el exp(coef) al ser un modelo 
# de regresion logistica

# Cociente de ventajas (Odd Ratio)
exp(coef(modelo_churn))

#  (Intercept)         EDAD      SEXOFem CIVILSoltero      HIJOS 
#    0.1590714    1.0100144   10.6414099    0.9604771  0.8813786 
#      INGRESO       AUTONo 
#    0.9999932    0.8646320 

# Para el caso de SEXO, el valor estimado 10.641 significa que, 
# manteniendo constantes el resto de las variables, 
# los clientes del genero FEMENINO tienen 10.641 veces mas 
# ventaja de FUGAR que los clientes que son del genero MASCULINO.

# Para el caso de la EDAD, ante un incremento en una unidad de 
# medida de la EDAD (un year), provocara un incremento 
# multiplicativo por un factor de 1.01 de la ventaja de FUGA 

cbind(Coeficientes=modelo_churn$coef,ExpB=exp(modelo_churn$coef))


# Cociente de ventajas e Intervalo de Confianza al 95% 
# Empleeanod library(MASS)
exp(cbind(OR = coef(modelo_churn),confint.default(modelo_churn)))


# Seleccion de Variables  
# library(MASS)
step <- stepAIC(modelo_churn,direction="backward", trace=FALSE)
step$anova
# Quedan 3 variables para describir el modelo
# Edad, Sexo e ingreso

# Modelo 2 con las variables m?s importantes
modelo_churn2 <- glm(CHURN ~ EDAD + SEXO + INGRESO, 
                     family=binomial,
                     data=training)

summary(modelo_churn2)
coef(modelo_churn2)

#     (Intercept)           EDAD         SEXOFem         INGRESO 
# -2.075173941837 0.009931234855  2.350304589387 -0.000006666961 


# Prediccion para nuevos individuos  

# Caso 1
nuevo1 <- data.frame(EDAD=57, SEXO="Fem",INGRESO=27535.30)
predict(modelo_churn2,nuevo1,type="response")

# Caso 2
nuevo2 <- data.frame(EDAD=40, SEXO="Masc",INGRESO=12535.50)
predict(modelo_churn2,nuevo2,type="response")

# Indicadores ====
# Para la evaluacion se usara el modelo_churn2 obtenido con la 
# muestra training y se evaluara en la muestra testing

# Prediciendo la probabilidad
proba.pred <- predict(modelo_churn2,testing,type="response")
head(proba.pred)

# Prediciendo la clase con punto de corte igual a 0.5
clase.pred <- ifelse(proba.pred >= 0.5, "Fuga","Actual")

head(clase.pred)

# Convirtiendo clase.pred a factor 
clase.pred <- as.factor(clase.pred)          

str(clase.pred)

head(cbind(testing,proba.pred,clase.pred),10)

write.csv(cbind(testing,proba.pred,clase.pred),
          "Testing con clase y proba predicha-Logistica.csv")

# Graficando la probabilidad predicha y la clase real
ggplot(testing, aes(x=proba.pred,color= CHURN,fill=CHURN)) + 
  geom_histogram(alpha = 0.25) + theme_bw()

# Tabla de clasificacion
# Calcular el % de acierto (accuracy)
accuracy <- mean(clase.pred==testing$CHURN)
accuracy


# Calcular el error de mala clasificaci?n
error <- mean(clase.pred!=testing$CHURN)
error

# Crosstable usando la siguiente libreria
# library(gmodels)
CrossTable(testing$CHURN,clase.pred,
           prop.t=FALSE, prop.c=FALSE,prop.chisq=FALSE)

# Usando el paquete caret
# library(caret)
cm <- caret::confusionMatrix(clase.pred,
                             testing$CHURN,
                             positive="Fuga")
cm

cm$table

cm$byClass["Sensitivity"] 
cm$byClass["Specificity"] 
cm$overall["Accuracy"]

# Empleando las metricas
# library(MLmetrics)
Precision(testing$CHURN, clase.pred,positive="Fuga")
Recall(testing$CHURN, clase.pred,positive="Fuga")
F1_Score(testing$CHURN, clase.pred,positive="Fuga")


# Curva ROC
# Usando el paquete caTools
# library(caTools)
AUC <- colAUC(proba.pred,testing$CHURN, plotROC = TRUE)
abline(0, 1,col="red") 

AUC  # Devuelve el ?rea bajo la curva