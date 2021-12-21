cat(format(Sys.time(), "%F %R"), ": READING ARGUMENTS\n")
args = commandArgs(trailingOnly = TRUE)
arch            <- args[2]
trainable       <- as.logical(args[3])
mod_name        <- args[4]
percentage      <- args[5]
selfmodel       <-
  ifelse(
    strsplit(mod_name, "_")[[1]][1] == "cpc",
    stringr::str_sub(mod_name, 13, 16),
    stringr::str_sub(mod_name, 1, 2)
  )
runname         <-
  paste(
    "gram",
    arch,
    "semi",
    stringr::str_sub(strsplit(mod_name, "_")[[1]][2], 1, 1),
    ifelse(selfmodel == "lm", "lm", "rn18"),
    ifelse(trainable, "ft", "fr"),
    percentage,
    "pct",
    sep = "_"
  )

trained_model <-
  paste("model_results/models/final/", mod_name, "/bestmod.h5", sep = "")

lrmin <- ifelse(trainable, 1e-7, 1e-5)
lrmax <- ifelse(trainable, 1e-4, 1e-2)

source("R/help/getenvBacDive.R")
source("R/help/supervised_help.R")
source("R/train.R")
source(paste0("R/architectures/supar/add_", arch, ".R"))

cat(format(Sys.time(), "%F %R"), ": PREPARING MODEL\n")
if (selfmodel != "lm") {
  model <- reduce_model(
    trainedmodel = trained_model,
    maxlen = 6700L,
    patchlen = 500L,
    trainable = trainable,
    enc = selfmodel
  )
} else{
  model <- reduce_model_LM(6700L, trained_model, trainable)
}

model1 <- addl(model, 2)
print("MODEL BUILT")

if (percentage != 100) {
    trpart <-
      round(mean(c(
        length(list.files(path[[1]])),
        length(list.files(path[[2]]))
      )) * as.numeric(percentage) / 100)
} else {
  trpart <- NULL
}
cat("Will use", trpart, "files per class.\n")

Train(
  #### data specs ####
  path,
  patchlen     = 6700L,
  step         = 6700L,
  randomFilesTrain = T,
  max_samples  = 16L,
  numberOfFiles = trpart,
  preloadGeneratorpath = paste0(preloadGeneratorpath, "gram"),
  ### model functions ###
  approach     = "supervised",
  task         = "gram",
  sup_nn_model = model1,
  ### hyperparameters general ###
  batch.size   = 32,
  epochs       = 100,
  steps.per.epoch = 3000,
  lr_schedule  = list(
    schedule   = "cosine_annealing",
    lrmin      = lrmin,
    lrmax      = lrmax,
    restart    = 20,
    mult       = 1.5
  ),
  k            = 1,
  #### callbacks ####
  run.name     = runname,
  tensorboard.log = tensorboard.log
)
