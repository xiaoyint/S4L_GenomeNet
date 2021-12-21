library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(tensorflow)

source("R/help/getenvBacDive.R")
source("R/help/supervised_help.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TRAIN DATA\n")
dattrain <-
  readRDS("/home/ru39qiq5/data/BacDive/fullmattrain.rds")
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TEST DATA\n")
dattest <-
  readRDS("/home/ru39qiq5/data/BacDive/fullmattest.rds")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET 10 PERCENT OF ROWS\n")
u <- data.table(dattrain[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_train <-
  sample(seq_along(dattrain[[1]]), length(dattrain[[1]]) / 10, prob = u$V1)
u <- data.table(dattest[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_test <-
  sample(seq_along(dattest[[1]]), length(dattest[[1]]) / 10, prob = u$V1)

dattrain10 <- list()
dattest10 <- list()
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "REMOVE OTHER ROWS\n")
dattrain10[[1]] <- dattrain[[1]][samp_train]
dattrain10[[2]] <- dattrain[[2]][samp_train]
dattest10[[1]] <- dattest[[1]][samp_test]
dattest10[[2]] <- dattest[[2]][samp_test]

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
saveRDS(dattrain10, "/home/ru39qiq5/data/BacDive/fullmattrain10.rds")
saveRDS(dattest10, "/home/ru39qiq5/data/BacDive/fullmattest10.rds")
rm(dattrain10)
rm(dattest10)

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET 1 PERCENT OF ROWS\n")
u <- data.table(dattrain[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_train <-
  sample(seq_along(dattrain[[1]]), (length(dattrain[[1]]) / 100), prob = u$V1)
u <- data.table(dattest[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_test <-
  sample(seq_along(dattest[[1]]), (length(dattest[[1]]) / 100), prob = u$V1)

dattrain1 <- list()
dattest1 <- list()
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "REMOVE OTHER ROWS\n")
dattrain1[[1]] <- dattrain[[1]][samp_train]
dattrain1[[2]] <- dattrain[[2]][samp_train]
dattest1[[1]] <- dattest[[1]][samp_test]
dattest1[[2]] <- dattest[[2]][samp_test]
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
saveRDS(dattrain1, "/home/ru39qiq5/data/BacDive/fullmattrain1.rds")
saveRDS(dattest1, "/home/ru39qiq5/data/BacDive/fullmattest1.rds")
rm(dattrain1)
rm(dattest1)


cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET 0.1 PERCENT OF ROWS\n")
u <- data.table(dattrain[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_train <-
  sample(seq_along(dattrain[[1]]), (length(dattrain[[1]]) / 1000), prob = u$V1)
u <- data.table(dattest[[2]])[, V1 := as.factor(as.numeric(V1))]
samp_test <-
  sample(seq_along(dattest[[1]]), (length(dattest[[1]]) / 1000), prob = u$V1)

dattrain01 <- list()
dattest01 <- list()
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "REMOVE OTHER ROWS\n")
dattrain01[[1]] <- dattrain[[1]][samp_train]
dattrain01[[2]] <- dattrain[[2]][samp_train]
dattest01[[1]] <- dattest[[1]][samp_test]
dattest01[[2]] <- dattest[[2]][samp_test]
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
saveRDS(dattrain01, "/home/ru39qiq5/data/BacDive/fullmattrain01.rds")
saveRDS(dattest01, "/home/ru39qiq5/data/BacDive/fullmattest01.rds")
rm(dattrain01)
rm(dattest01)

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "DONE\n")
