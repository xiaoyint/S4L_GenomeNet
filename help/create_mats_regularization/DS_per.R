library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(tensorflow)

source("R/help/getenvdeepSea.R")
source("R/help/supervised_help.R")

for (i in c(10, 1, 0.1, 0.01)) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TRAIN DATA\n")
  dattrain <- readRDS(paste0(path, "/fullmattrain.rds"))
  
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"),
    "GET", i, "PERCENT OF ROWS\n")
  samp_train <-
    sample(length(dattrain[[1]]), (length(dattrain[[1]]) * i/100))
  
  datper <- list()
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"),
    "REMOVE OTHER ROWS\n")
  datper[[1]] <- dattrain[[1]][samp_train]
  datper[[2]] <- dattrain[[2]][samp_train]
  
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
  saveRDS(datper, paste0(path, "/fullmattrain", i, ".rds"))
  rm(datper)
}

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "DONE\n")
