args = commandArgs(trailingOnly = TRUE)
spec  <- args[1]
if (spec == "bacteria") {
  species <- list("bacteria_1_3")
} else {
  species <- list("human_0_1", "bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
}

source("R/help/getenv.R")
source("R/train.R")

run.name <- paste("lm", spec, sep = "_")

source("R/architectures/LM_danQ.R")
model <- danQ(500L)

Train(
  #### data specs ####
  path             = path,
  path.val         = path.val,
  maxlen           = 500L,
  step             = 500L,
  max_samples      = 16,
  randomFilesTrain = TRUE,
  preloadGeneratorpath = paste0(preloadGeneratorpath, spec),
  #### model specs ####
  ### model functions ###
  approach         = "self-supervised-nextnuc",
  sup_nn_model     = model,
  ### hyperparameters general ###
  batch.size       = 512,
  epochs           = 200,
  steps.per.epoch  = 1010,
  validation_split = 0.2,
  lr_schedule      = list(
         schedule  = "cosine_annealing",
            lrmin  = 1e-4,
            lrmax  = 1e-2,
          restart  = 20,
             mult  = 1.5),
  k                = 1,
  #### callbacks ####
  run.name         = run.name,
  tensorboard.log  = tensorboard.log
)
