species <- list("bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
source("R/help/getenv.R")
suppressMessages(library(deepG))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))
suppressMessages(library(reticulate))
suppressMessages(library(data.table))
suppressMessages(library(glmnet))
suppressMessages(library(hdf5r))
library(stringi)

cat(format(Sys.time(), "%F %R"), ": Joining Test data matrices\n")
amtest <- list()
for (i in 1:3) {
  amtest[[i]] <- readRDS(paste0(sub("\\/train\\/", "", path[[i]]), "/fullmattest.rds"))
  amtest[[i]][[3]] <- list(rep(i, length(amtest[[i]][[1]])))
}
testdatall <- list(c(amtest[[1]][[1]], amtest[[2]][[1]], amtest[[3]][[1]]),
                   c(amtest[[1]][[2]], amtest[[2]][[2]], amtest[[3]][[2]]),
                   c(amtest[[1]][[3]][[1]], amtest[[2]][[3]][[1]], amtest[[3]][[3]][[1]]))
cat(format(Sys.time(), "%F %R"), ": Saving\n")
saveRDS(testdatall, paste0(savepath, "fullmattest.rds"))
rm(amtest, testdatall)
cat(format(Sys.time(), "%F %R"), ": Test done\n")

cat(format(Sys.time(), "%F %R"), ": Joining Val data matrices\n")
amval <- list()
for (i in 1:3) {
  amval[[i]] <- readRDS(paste0(sub("\\/train\\/", "", path[[i]]), "/fullmatval.rds"))
  amval[[i]][[3]] <- list(rep(i, length(amval[[i]][[1]])))
}
valdatall <- list(c(amval[[1]][[1]], amval[[2]][[1]], amval[[3]][[1]]),
                  c(amval[[1]][[2]], amval[[2]][[2]], amval[[3]][[2]]),
                  c(amval[[1]][[3]][[1]], amval[[2]][[3]][[1]], amval[[3]][[3]][[1]]))
cat(format(Sys.time(), "%F %R"), ": Saving\n")
saveRDS(valdatall, paste0(savepath, "fullmatval.rds"))
rm(amval, valdatall)
cat(format(Sys.time(), "%F %R"), ": Val done\n")

cat(format(Sys.time(), "%F %R"), ": Joining Train data matrices\n")
amtrain <- list()
for (i in 1:3) {
  amtrain[[i]] <- readRDS(paste0(sub("\\/train\\/", "", path[[i]]), "/fullmattrain.rds"))
  amtrain[[i]][[3]] <- list(rep(i, length(amtrain[[i]][[1]])))
}
traindatall <- list(c(amtrain[[1]][[1]], amtrain[[2]][[1]], amtrain[[3]][[1]]),
                    c(amtrain[[1]][[2]], amtrain[[2]][[2]], amtrain[[3]][[2]]),
                    c(amtrain[[1]][[3]][[1]], amtrain[[2]][[3]][[1]], amtrain[[3]][[3]][[1]]))
cat(format(Sys.time(), "%F %R"), ": Saving\n")
saveRDS(traindatall, paste0(savepath, "fullmattrain.rds"))
rm(amtrain, traindatall)
cat(format(Sys.time(), "%F %R"), ": Train done\n")