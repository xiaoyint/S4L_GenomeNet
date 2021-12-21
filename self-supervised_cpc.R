args = commandArgs(trailingOnly = TRUE)
spec        <- args[1]
enc         <- args[2]
max_samples <- as.integer(args[3])

run.name <- paste(enc, max_samples, spec, sep = "_")

if (spec == "bacteria") {
  species <- list("bacteria_1_3")
} else {
  species <- list("human_0_1", "bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
}

lr_schedule <- list(
  schedule       = "cosine_annealing",
  lrmin          = 1e-6,
  lrmax          = 1e-2,
  restart        = 50,
  mult           = 1.5
)
lr_schedule_shift <- 48

source(paste0("R/architectures/enc_", enc, ".R"), local = T)
source(paste0("R/architectures/ctx.R"), local = T)

source("R/help/getenv.R")
source("R/architectures/cpcloss.R")
source("R/train.R")

Train(
  #### data specs ####
  path              = path,
  path.val          = path.val,
  patchlen          = 500L,
  nopatches         = 32L,
  max_samples       = max_samples,
  randomFilesTrain  = TRUE,
  randomFilesVal    = TRUE,
  preloadGeneratorpath = paste0(preloadGeneratorpath, spec),
  #### model specs ####
  ### model functions ###
  encoder           = encoder,
  context           = context,
  cpcloss           = cpc,
  seed              = seed,
  ### hyperparameters general ###
  batch.size        = 32,
  epochs            = 300,
  steps.per.epoch   = 3000,
  validation_split  = 0.2,
  lr_schedule       = lr_schedule,
  lr_schedule_shift = lr_schedule_shift,
  emb_scale         = 0.001,
  k                 = 1,
  #### callbacks ####
  run.name          = run.name,
  tensorboard.log   = tensorboard.log
)
