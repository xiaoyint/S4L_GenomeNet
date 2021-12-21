trained_model   <- NULL
args = commandArgs(trailingOnly = TRUE)
selfmodel       <- args[1]  # one of "rn50", "rn18", "danQ"
arch            <- args[2]
percentage      <- args[3]
runname <-
  paste("gram_bl", selfmodel, arch, percentage, "pct", sep = "_")

source("R/help/getenvBacDive.R")
source("R/help/supervised_help.R")
source("R/train.R")
source(paste0("R/architectures/supar/add_", arch, ".R"))

if (selfmodel %in% c("rn50", "rn18", "danQ")) {
  model <- create_sup_model(maxlen = 6700L,
    patchlen = 500L,
    enc = selfmodel)
} else if (selfmodel == "lm") {
  model <- create_sup_modelLM(6700L)
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
  patchlen          = 500L,
  step         			= 6700L,
  randomFilesTrain 	= T,
  max_samples  			= 16L,
  numberOfFiles 		= trpart,
  preloadGeneratorpath 	= paste0(preloadGeneratorpath, "gram"),
  ### model functions ###
  approach     			= "supervised",
  task         			= "gram",
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