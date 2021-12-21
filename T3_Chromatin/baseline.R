trained_model   <- NULL
args = commandArgs(trailingOnly = TRUE)
selfmodel       <- args[1] # one of "rn50", "rn18", "danQ"
arch            <- args[2]
percentage      <- as.integer(args[3])

runname <- paste("chrom_bl", selfmodel, arch, percentage, "pct", sep = "_")

source("R/help/getenvdeepSea.R")
source("R/help/supervised_help.R")
source("R/train.R")
source(paste0("R/architectures/supar/add_", arch, ".R"))

if (selfmodel %in% c("rn50", "rn18", "danQ")) {
  model <- create_sup_model(maxlen = 900L,
    patchlen = 500L,
    enc = selfmodel)
} else if (selfmodel == "lm") {
  model <- create_sup_modelLM(900L)
}

model1 <- addl(model, 919, "sigmoid")

print("MODEL BUILT")

if (percentage != 100) { 
  pathtrain <- paste0(sub("deepsea_train\\/.*","", path),"subset_", percentage, "_perc.rds")
} else {
  pathtrain <- paste0(path, "train")
}

Train(
  #### data specs ####
  pathtrain,
  patchlen     = 1000L,
  step         = 1000L,
  preloadGeneratorpath = preloadGeneratorpath,
  ### model functions ###
  approach     = "supervised",
  sup_nn_model = model1,
  task         = "deepSea",
  ### hyperparameters general ###
  batch.size   = 32,
  epochs       = 400,
  steps.per.epoch = 3000,
  lr_schedule  = list(
    schedule   = "cosine_annealing",
    lrmin      = 1e-6,
    lrmax      = 1e-2,
    restart    = 50,
    mult       = 1.5
  ),
  #### callbacks ####
  run.name = runname,
  tensorboard.log = tensorboard.log
)
