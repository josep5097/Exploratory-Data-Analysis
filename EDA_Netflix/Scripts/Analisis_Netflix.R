# Casos de Netflix

# Librerias
library(dplyr)
library(ggplot2)
library(corrplot)

# Data ====
DB <- read.csv("Data/netflix.csv")


# Manipulacion ====
str(DB)
head(DB)
# Resumen
summary(DB)
# Eliminación de casos NA
DB1 <- na.omit(DB)
str(DB1)
summary(is.na(DB))
attach(DB1)

pairs(~ DB1$ratingDescription+
        DB1$release.year+
        DB1$user.rating.score+
        DB1$user.rating.size)
corr1 <- cor(DB1[,4:7])
corrplot(corr1,
         method = 'pie',
         title = "Correlación entre variables" )
