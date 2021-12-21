cat(format(Sys.time(), "%F %R"), ": READING ARGUMENTS\n")
args = commandArgs(trailingOnly = TRUE)
arch            <- args[1]
trainable       <- as.logical(args[2])
mod_name        <- args[3]
percentage      <- as.integer(args[4])
selfmodel       <- ifelse(strsplit(mod_name, "_")[[1]][1] == "cpc", stringr::str_sub(mod_name, 13, 16), stringr::str_sub(mod_name, 1, 2))
runname         <- paste("chrom", arch, "semi", stringr::str_sub(strsplit(mod_name, "_")[[1]][2],1,1), ifelse(selfmodel == "lm", "lm", "rn18"), ifelse(trainable, "ft", "fr"), percentage, "pct", sep = "_")

trained_model <- paste("model_results/models/final/", mod_name, "/bestmod.h5", sep = "")

lrmin <- ifelse(trainable, 1e-7, 1e-5)
lrmax <- ifelse(trainable, 1e-4, 1e-2)

source("R/help/getenvdeepSea.R")
source("R/help/supervised_help.R")
source("R/train.R")
source(paste0("R/architectures/supar/add_", arch, ".R"))

cat(format(Sys.time(), "%F %R"), ": PREPARING MODEL\n")
if (selfmodel != "lm") {
  model <- reduce_model(
    trainedmodel = trained_model,
    maxlen = 900L,
    patchlen = 500L,
    trainable = trainable,
    enc = selfmodel
  )
} else{
  model <- reduce_model_LM(900L, trained_model, trainable)
}

model1 <- addl(model, 919, "sigmoid")
 
if (percentage != 100) { 
  pathtrain <- paste0(sub("deepsea_train\\/.*","", path),"subset_", percentage, "_perc.rds")
} else {
  pathtrain <- paste0(path, "train")
}

cat(format(Sys.time(), "%F %R"), ": START TRAINING\n")
Train(
  #### data specs ####
  pathtrain,
  patchlen     = 1000L,
  step         = 1000L,
  preloadGeneratorpath = preloadGeneratorpath,
  ### model functions ###
  approach     = "supervised",
  task         = "deepSea",
  sup_nn_model = model1,
  ### hyperparameters general ###
  batch.size   = 32,
  epochs       = 200,
  steps.per.epoch = 3000,
  lr_schedule  = list(
    schedule   = "cosine_annealing",
    lrmin      = lrmin,
    lrmax      = lrmax,
    restart    = 20,
    mult       = 1.5
  ),
  #### callbacks ####
  run.name     = runname,
  tensorboard.log = tensorboard.log
)
