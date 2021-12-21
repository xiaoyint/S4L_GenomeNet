########################################################################################################
############################################## PREPARATION #############################################
########################################################################################################

cat(format(Sys.time(), "%F %R"), ": PREPARE VARIABLES\n")

library(data.table)
library(deepG)
library(dplyr)
library(keras)
library(mlr3)
library(mlr3learners)
require(bbotk) # terminator
library(tensorflow)
library(future.apply)
library(abind)

args = commandArgs(trailingOnly = TRUE)
mod_name      <- args[1]
percentage    <- as.double(args[2])

selfmodel     <-
  ifelse(
    strsplit(mod_name, "_")[[1]][1] == "cpc",
    stringr::str_sub(mod_name, 13, 16),
    stringr::str_sub(mod_name, 1, 2)
  )
name          <-
  ifelse(
    selfmodel == "lm",
    paste(
      stringr::str_sub(mod_name, 4, 6),
      stringr::str_sub(mod_name, 1, 2),
      sep = "_"
    ),
    paste(
      stringr::str_sub(mod_name, 5, 7),
      stringr::str_sub(mod_name, 13, 19),
      sep = "_"
    )
  )
runname       <-
  paste0(name, "_", percentage, format(Sys.time(), "_%y%m%d_%H%M%S"))

trained_model <-
  paste("model_results/models/final/", mod_name, "/bestmod.h5", sep = "")
dir.create(paste0("ssl_results/gram_", runname))

source("R/help/getenvBacDive.R")
source("R/help/supervised_help.R")

########################################################################################################
########################################### DATA PREPARATION ###########################################
########################################################################################################

cat(format(Sys.time(), "%F %R :"), "LOAD ", percentage, "% TRAIN DATA\n")
if (percentage == 100) {
  dattrain <- readRDS(paste0(savepath, "fullmattrain.rds"))
} else{
  dattrain <-
    readRDS(paste0(savepath, "fullmattrain", percentage, ".rds"))
}
cat(format(Sys.time(), "%F %R :"), percentage, "% TRAIN DATA LOADED\n")

rowno <- length(dattrain[[1]])
cat(format(Sys.time(), "%F %R :"), rowno, "STATES TO CREATE\n")
divno <- round(rowno / 5)
split <- seq(0, rowno, divno)
split[6] <- rowno
samp <- lapply(1:5, function(x) {
  c((split[[x]] + 1):split[[x + 1]])
})

p <- 1:5
dt <- list()
for (i in p) {
  dt[[i]] <- list(dattrain[[1]][samp[[i]]], dattrain[[2]][samp[[i]]])
}
rm(dattrain)

cat(format(Sys.time(), "%F %R :"), "START PARALLEL COMPUTATION\n")
future::plan("multisession", workers = 5)
a3 <- future_lapply(dt, function(X) {
  tf <- import('tensorflow')
  Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)
  tf$config$threading$set_intra_op_parallelism_threads(1L)
  tf$config$threading$set_inter_op_parallelism_threads(1L)
  if (selfmodel == "lm") {
    model <- reduce_model_LM(6700L, trained_model, F)
    outsize <- 320
  } else {
    model <- reduce_model(
      trainedmodel = trained_model,
      maxlen = 6700L,
      patchlen = 500L,
      trainable = F,
      enc = selfmodel
    )
    outsize <- model$output$shape[2][[1]]
  }
  
  getstatesmat(model, X, 1, outsize)
}, future.seed = T, future.packages = c("deepG", "dplyr", "tensorflow"))
cat(format(Sys.time(), "%F %R :"), "PARALLEL COMPUTATION FINISHED\n")
future::plan("sequential")

fin <- list()
fin[[1]] <-
  abind(lapply(p, function(X) {
    a3[[X]][1][[1]]
  }), along = 1)
fin[[2]] <-
  abind(lapply(p, function(X) {
    a3[[X]][2][[1]]
  }), along = 1)

dtr <- data.table(fin[[1]], target = as.factor(fin[[2]]))
m2 = list(
  msr("classif.auc"),
  msr("classif.acc"),
  msr("classif.bacc"),
  msr("classif.fbeta"),
  msr("classif.logloss")
)

cat(format(Sys.time(), "%F %R :"), nrow(dtr), "STATES CREATED\n")

cat(format(Sys.time(), "%F %R :"), "SAVE STATES\n")
saveRDS(dtr,
  paste0("ssl_results/gram_", runname, "/L1_statesTR.rds"))
cat(format(Sys.time(), "%F %R :"), "STATES SAVED\n")

########################################################################################################
################################################ GLMNET ################################################
########################################################################################################

cat(format(Sys.time(), "%F %R :"), "CREATE GLMNET\n")

# define task
task = TaskClassif$new(id = "cv_glmnet",
  backend = dtr,
  target = "target")

# create learner
lnr <-
  lrn(
    "classif.cv_glmnet",
    alpha = 1,
    type.measure = "deviance",
    predict_type = "prob",
    predict_sets = c("train", "test"),
    trace.it = 1
  )

# stratified resampling for the target
task$col_roles$stratum <- "target"

cat(format(Sys.time(), "%F %R :"), "START TRAINING\n")
lnr$train(task)

lnr$model
# Save result
saveRDS(lnr,
  file = paste0("ssl_results/gram_", runname, "/L1_learner.rds"))

########################################################################################################
############################################## PREDICTION ##############################################
########################################################################################################

cat(format(Sys.time(), "%F %R :"), "READ TEST STATES\n")
dte <-
  readRDS(paste0(preloadGeneratorpath, "states/gram/", "test_", name, ".rds"))
cat(format(Sys.time(), "%F %R :"), "TEST STATES READ\n")

# See result
lnr$predict(task = task)$confusion

cat(format(Sys.time(), "%F %R :"), "PREDICT ON TRAIN/TEST DATA\n")
prtr <- lnr$predict_newdata(newdata = dtr, task = task)
prte <- lnr$predict_newdata(newdata = dte, task = task)

results <- list(
  "Model" = trained_model,
  "TrainConfusion" = prtr$confusion,
  "TrainMeasures" = prtr$score(measures = m2),
  "TrainF1"      = f1_from_conf_matrix(prtr$confusion),
  "TestConfusion" = prte$confusion,
  "TestMeasures" = prte$score(measures = m2),
  "TestF1" = f1_from_conf_matrix(prte$confusion)
)

results

sink(paste0("ssl_results/gram_", runname, "/L1_results.txt"))
print(results)
sink()
