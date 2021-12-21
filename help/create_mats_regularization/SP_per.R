library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(stringi)
library(stringr)
library(tensorflow)

species <- list("bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
source("R/help/getenv.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "READ TRAIN DATA\n")
dattrain <- readRDS(paste0(savepath, "fullmattrain.rds"))
w <- data.table(file = unlist(dattrain[[2]]), label = unlist(dattrain[[3]]))
w$label        <- as.factor(w$label)
bacteria       <- w[label == 1,]
viral_phage    <- w[label == 2,]
viral_no_phage <- w[label == 3,]
samp_ba        <- levels(as.factor(bacteria$file))
samp_vp        <- levels(as.factor(viral_phage$file))
samp_vn        <- levels(as.factor(viral_no_phage$file))
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "With 100% we have", length(samp_ba), 
  "files for bacteria,", length(samp_vp), "files for viral-phage, and",  length(samp_vn), 
  "files for viral-no-phage.\n")

for (i in c(10, 1, 0.1)) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "GET", i, "PERCENT OF ROWS\n")
  samp_ba <- sample(samp_ba, round(length(samp_ba) * 0.1))
  samp_vp <- sample(samp_vp, round(length(samp_vp) * 0.1))
  samp_vn <- sample(samp_vn, round(length(samp_vn) * 0.1))
  samp <- which(w$file %in% c(samp_ba, samp_vp, samp_vn))
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S : With"), i , "% we have", length(samp_ba), "files for bacteria,", 
    length(samp_vp), "files for viral-phage, and",  length(samp_vn), "files for viral-no-phage. In total we have", 
    length(samp), "sequences\n")
  
  datper <- list()
  datper[[1]] <- dattrain[[1]][samp]
  datper[[2]] <- dattrain[[3]][samp]

  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVE\n")
  saveRDS(datper, paste0(savepath, "/fullmattrain", i, ".rds"))
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVING FINISHED\n")
}

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "PERCENTAGES DONE\n")
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "CREATE 5 PARTS OF 100 % DATA FOR SPEED\n")

rowno <- length(dattrain[[1]])
divno <- round(rowno/10)
split <- seq(0, rowno, divno)
split[11] <- rowno
samp <- lapply(1:10, function(x) {
  c((split[[x]] + 1):split[[x + 1]])
})

p <- 1:10
for (j in p) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "SAVING PARTS", j, "\n")
  saveRDS(list(dattrain[[1]][samp[[j]]], dattrain[[3]][samp[[j]]]), paste0(savepath, "/fullmattrain_pt", j, ".rds"))
}
