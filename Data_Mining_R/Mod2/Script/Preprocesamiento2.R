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
library(rpart)
library(rpart.plot)

# Revisar los tiempos al cargar datos 
tic()
census.csv <- read.csv("censusn.csv",sep=";")
toc()

# Cargando los datos con la funcion fread() de data.table
tic()
censusn <-fread("Mod2/Data/censusn.csv", 
                header=T, 
                verbose =FALSE, 
                stringsAsFactors=TRUE,
                showProgress =TRUE)
toc()

str(censusn)
# Quitamos ID
censusn$id <- NULL

# Tratamiento de los datos ====
# 1. Deteccion de valores perdidos

# Deteccion de valores perdidos con el paquete DataExplorer
plot_missing(censusn) 



# 2. Eliminacion de datos perdidos
census.cl <- na.omit(censusn)

plot_missing(census.cl) 

# 3. Imputacion con el paquete DMwR ====
# Data Mining with R, Luis Torgo

# Funcion centralImputation()
# Si la variable es num?rica (numeric o integer) reemplaza los
# valores faltantes con la mediana.
# Si la variable es categorica (factor) reemplaza los valores 
# faltantes con la moda. 

census.ci <-centralImputation(censusn)
plot_missing(census.ci) 


# 4. Imputar los datos usando k-nn
# Emplearlo para var numerica y categorica
# Los vuelve a la escala original
set.seed(123)
censusn_transformado3 <- knnImputation(censusn,
                                       scale = TRUE, 
                                       k=3)

# Verificando la cantidad de valores perdidos
sum(is.na(censusn_transformado3))

plot_missing(censusn_transformado3)

# Deteccion de outliers univariados ====
# Para la variable age
ggplot(censusn, 
       aes(x="",age)) + 
  geom_boxplot(fill="peru") +
  labs(x="")

boxplot(censusn$age,
        col="peru")
# El grafico de boxplot me indica los valores outliers
outliers1 <- boxplot(censusn$age,
                     col="peru")$out
outliers1 ; length(outliers1) 
summary(outliers1)

# Filtrando los valores outliers de age
censusn.out <- censusn %>% 
  filter(age<79)

ggplot(censusn.out, 
       aes(x="",age)) +
  geom_boxplot(fill="peru") +
  labs(x="")
# Se tienen nuevos outliers
outliers2 <- boxplot(censusn.out$age,
                     col="cadetblue3")$out
outliers2 ; length(outliers2)
summary(outliers2)
# La solucion no es eliminar, sino discretizar los valores numericos

# Para la variable hours.per.week
ggplot(censusn, aes(x="",hours.per.week)) + 
       geom_boxplot(fill="lightgreen") +
       labs(title="Boxplot de Hours.per.week",
            x="") +
       theme_minimal()

outliers3 <- boxplot(censusn$hours.per.week)$out
outliers3 ; length(outliers3)
summary(outliers3)
# Se tiene un margen amplio en los valores numericos

# Discretizar los valores ====
# Discretizacion usando BoxPlot y la funcion cut()

censusd4 <- censusn

summary(censusd4$hours.per.week)

ggplot(aes(x = "", 
           y = hours.per.week),
       data = censusd4) + 
  geom_boxplot() +
  scale_y_continuous(breaks = seq(0, 100, 5)) + 
  labs(title="Box Plot de horas de trabajo a la semana",
       xlab="", ylab = "Horas de trabajo a la semana") +
  theme_minimal()

summary(censusd4$hours.per.week)

# Menos de 40
# De 40 a 45
# De 45 a mas

censusd4$hpw_cat4 <- cut(censusd4$hours.per.week, 
                         breaks = c(-Inf,40,45,Inf),
                         labels = c("Menos de 40", 
                                    "De 40 a menos de 45",
                                    "De 45 a mas"),
                         right = FALSE)  # [   >

table(censusd4$hpw_cat4)  

ggplot(censusd4, 
       aes(hpw_cat4)) + 
  geom_bar(color="black",
           fill="darkgreen") + 
  theme_light() + 
  labs(title ="Grafico de Barras", 
       x="Horas de trabajo a la semana", 
       y= "Frecuencia") 

# Verificando con un grafico de barras apilado
ggplot(censusd4, 
       aes(x= hpw_cat4,
           fill=salary ) ) +
  geom_bar(position=position_fill()) +
  theme_bw() +
  labs(title ="Salario segun las horas de trabajo a la semana",
       x="Horas de trabajo", 
       y= "Frecuencia") + 
  scale_fill_manual(values=c("darkolivegreen3", 
                             "firebrick2")) +
  theme(legend.position="bottom") 


# Discretizacion usando arboles de clasificacion
# Empleando rpart
censusd6 <- censusn
set.seed(123)
arbol <- rpart(salary ~  hours.per.week,
                        data=censusd6,
                        method="class",  # anova
                        control=rpart.control(cp=0,minbucket=0)
              )

rpart.plot(arbol, 
           digits=-1,
           type=2, 
           extra=101,
           varlen = 3,
           cex = 0.7, 
           nn=TRUE)
# Se toman los valores del punto de corte del arbol para generar las segmentaciones
# para discretizar los valores.
# Menos de 42
# De 42 a mas

censusd6$hpw_cat6 <- cut(censusd6$hours.per.week, 
                         breaks = c(-Inf,42,Inf),
                         labels = c("Menos de 42",
                                    "De 42 a mas"),
                         right = FALSE)

table(censusd6$hpw_cat6)  
prop.table(table(censusd6$hpw_cat6,
                 censusd6$salary),
           margin=1)

ggplot(censusd6, 
       aes(hpw_cat6)) + 
  geom_bar(color="black",
           fill="orange") + 
  theme_light() + 
  labs(title ="Grafico de Barras", 
       x="Horas de trabajo a la semana", 
       y= "Frecuencia") 

# Verificando con un grafico de barras apilado
ggplot(censusd6, 
       aes(x= hpw_cat6,
           fill=salary ) ) +
  geom_bar(position=position_fill()) +
  theme_bw() +
  labs(title ="Salario segun las horas de trabajo a la semana",
       x="Horas de trabajo", 
       y= "Frecuencia") + 
  scale_fill_manual(values=c("darkolivegreen3", 
                             "firebrick2")) +
  theme(legend.position="bottom") 
