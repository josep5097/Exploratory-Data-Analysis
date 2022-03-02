# Librerias ====
library(openxlsx)
library(foreign)
library(forecast)
library(gmodels)
library(lmtest)
library(ResourceSelection)
library(ROCR)

#cargar la base de datos ====
DB_1 <- read.csv("Data/Telco-Customer-Churn.csv",
                 header = T,
                 sep = ",",
                 stringsAsFactors = F)
