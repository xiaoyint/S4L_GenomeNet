library(deepG)
host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)
lochelm <- "/projects/BIFO/genomenet/training_data/"
species <- list("bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  savepath        <- paste0("/vol", lochelm)
  tensorboard.log <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
  preloadGeneratorpath <-  "/vol/projects/BIFO/genomenet/preprocessed_data/"
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  savepath        <- paste0("/net", lochelm)
  tensorboard.log <- "/net/projects/BIFO/genomenet/tensorboard/yin"
  preloadGeneratorpath <-  "/net/projects/BIFO/genomenet/preprocessed_data/"
} else if (host == "mcml-d") {
  setwd("/home/ru39qiq5/MAGenomeNet")
  savepath        <- "/home/ru39qiq5/genedata/"
  tensorboard.log <- "/home/ru39qiq5/tensorboard"
  preloadGeneratorpath <- "/home/ru39qiq5/genedata/preprocessed_data/"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  savepath <- "/genedata/genomes/"
  tensorboard.log <- "/home/user/tensorboard"
}
path.val        <- as.list(paste0(savepath, species, "/validation/"))

source("R/help/preloadedGenerator.R")
bs <- 32

for (ml in c(6700, 7000, 7600)) {
  for (seed in c(1234, 111, 222, 888, 999)) {
    cat(format(Sys.time(), "%F %R"),
      ": Creating Generator Seed",
      seed, "maxlen", ml, 
      "\n")
    initializeGenerators(
      path.val,
      batch.size = 33,
      maxlen = ml,
      randomFiles = T,
      step = ml,
      seed = seed,
      max_samples = 8
    )
    
    preloadPLG2(
      labelByFolderGeneratorWrapper,
      1200,
      paste0(
        preloadGeneratorpath, 
        "species/batchsize",
        bs,
        "_maxlen",
        ml,
        "_max_samples8_seed_",
        seed,
        ".rds"
      ),
      F, bs, NULL, 33, path.val, NULL, ml
    )
    cat(format(Sys.time(), "%F %R"), ": FINISHED\n")
  }
}