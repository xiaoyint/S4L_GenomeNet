library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(stringi)
library(stringr)
library(tensorflow)

source("R/help/getenvBacDive.R")
source("R/help/supervised_help.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TRAIN DATA\n")
dattrain <- readRDS(paste0(sub("\\/datagram/negative\\/.*", "", path)[[1]], "/datagram/fullmattrain.rds"))
w <- data.table(file = unlist(dattrain[[2]]), label = unlist(dattrain[[3]]))
w$label <- as.factor(w$label)
neg <- w[label == 1,]
pos <- w[label == 2,]
samp_pos <- levels(as.factor(pos$file))
samp_neg <- levels(as.factor(neg$file))
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "With 100% we have", length(samp_pos), "files for gram positive and", 
  length(samp_neg), "files for gram negative.\n")

for (i in c(10, 1, 0.1)) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET", i, "PERCENT OF ROWS\n")
  samp_pos <- sample(samp_pos, round(length(samp_pos) * 0.1))
  samp_neg <- sample(samp_neg, round(length(samp_neg) * 0.1))
  samp <- which(w$file %in% c(samp_pos, samp_neg))
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S : With"), i , "% we have", length(samp_pos), "files for gram positive and", 
    length(samp_neg), "files for gram negative. In total we have", length(samp), "sequences\n")
  
  datper <- list()
  datper[[1]] <- dattrain[[1]][samp]
  datper[[2]] <- dattrain[[3]][samp] 
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE", i, "PERCENT DATA\n")
  saveRDS(datper, paste0(sub("\\/datagram/negative\\/.*", "", path)[[1]], "/datagram/fullmattrain", i,".rds"))
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVING FINISHED\n")
  #rm(samp_pos, samp_neg, samp, datper)
}

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "PERCENTAGES DONE\n")
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "CREATE 5 PARTS OF 100 % DATA FOR SPEED\n")

rowno <- length(dattrain[[1]])
divno <- round(rowno/5)
split <- seq(0, rowno, divno)
split[6] <- rowno
samp <- lapply(1:5, function(x) {
  c((split[[x]] + 1):split[[x + 1]])
})

p <- 1:5
for (j in p) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVING PARTS", j, "\n")
  saveRDS(list(dattrain[[1]][samp[[j]]], dattrain[[3]][samp[[j]]]), 
    paste0(sub("\\/datagram/negative\\/.*", "", path)[[1]], "/datagram/fullmattrain_pt", j,".rds"))
}

