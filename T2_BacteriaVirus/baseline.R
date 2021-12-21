trained_model   <- NULL
args = commandArgs(trailingOnly = TRUE)
selfmodel       <- args[1]  # one of "rn50", "rn18", "danQ"
arch            <- args[2]
percentage      <- args[3]
runname <-
  paste("BacVir_bl", selfmodel, arch, percentage, "pct", sep = "_")

species <-
  list("bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")


source("R/help/getenv.R")
source("R/help/supervised_help.R")
source("R/train.R")
source(paste0("R/architectures/supar/add_", arch, ".R"))

if (selfmodel %in% c("rn50", "rn18", "danQ")) {
  model <- create_sup_model(6700L = 6700L,
    patchlen = 500L,
    enc = selfmodel)
} else if (selfmodel == "lm") {
  model <- create_sup_modelLM(6700)
}

model1 <- addl(model, 3)
print("MODEL BUILT")

if (percentage != 100) {
  trpart <-
    round(mean(c(
      length(list.files(path[[1]])),
      length(list.files(path[[2]])),
      length(list.files(path[[3]]))
    )) * as.numeric(percentage) / 100)
} else {
  trpart <- NULL
}
cat("Will use", trpart, "files per class.\n")

Train(
  #### data specs ####
  path,
  patchlen          = 6700L,
  step         			= 6700L,
  randomFilesTrain 	= T,
  max_samples  			= 16L,
  numberOfFiles 		= trpart,
  preloadGeneratorpath 	= paste0(preloadGeneratorpath, "BacVir"),
  ### model functions ###
  approach     			= "supervised",
  task         			= "BacVir",
  sup_nn_model 			= model1,
  ### hyperparameters general ###
  batch.size   			= 32,
  epochs       			= 200,
  steps.per.epoch 	= 3000,
  lr_schedule  			= list(
    schedule   			= "cosine_annealing",
    lrmin      			= 1e-6,
    lrmax      			= 1e-2,
    restart    			= 50,
    mult       			= 1.5
  ),
  k           			= 1,
  #### callbacks ####
  run.name     			= runname,
  tensorboard.log 	= tensorboard.log
)