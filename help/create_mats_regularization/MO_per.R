library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(tensorflow)

source("R/help/getenvMotility.R")
source("R/help/supervised_help.R")

for (i in c(10, 1, 0.1)) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TRAIN DATA\n")
  dattrain <- readRDS(paste0(sub("\\/data\\/.*", "", path), "/MOT/fullmattrain.rds"))
  
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET", i, "PERCENT OF ROWS\n")
  w <- (rbindlist(dattrain[[3]]))
  w$V1 <- as.factor(w$V1)
  samp_train <- sample(levels(w$V1), (length(levels(w$V1)) * i/100))
  samp <- which(w$V1 %in% samp_train)
  
  datper <- list()
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"),
    "REMOVE OTHER ROWS\n")
  datper[[1]] <- dattrain[[1]][samp]
  datper[[2]] <- dattrain[[2]][samp]
  
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
  saveRDS(datper, paste0(sub("\\/data\\/.*", "", path), "/MOT/fullmattrain", i, ".rds"))
  rm(datper)
}

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "DONE\n")
