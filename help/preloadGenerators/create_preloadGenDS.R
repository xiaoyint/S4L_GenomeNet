library(deepG)
host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  path                 <- "/vol/projects/BIFO/genomenet/training_data/deepSea/deepsea_train/validation/"
  preloadGeneratorpath <- "/vol/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log      <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  path                 <- "/net/projects/BIFO/genomenet/training_data/deepSea/deepsea_train/validation/"
  preloadGeneratorpath <- "/net/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log      <- "/net/projects/BIFO/genomenet/tensorboard/yin"
} else if (host %in% c("mcml-d", "gpu-00", "dgx-00", "datala")) {
  setwd("/home/ru39qiq5/MAGenomeNet")
  path                 <- "/home/ru39qiq5/genedata/deepSea/validation/"
  preloadGeneratorpath <- "/home/ru39qiq5/genedata/preprocessed_data/"
  tensorboard.log      <- "/home/ru39qiq5/tensorboard"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  path                 <- "/genedata/genomes/deepSea/deepsea_train/validation/"
  preloadGeneratorpath <-  "/genedata/genomes/preprocessed_data/"
  tensorboard.log      <- "/home/user/tensorboard"
}

source("R/help/preloadedGenerator.R")
bs <- 32

for (seed in c(1234, 111, 222, 888, 999)) {
  cat(format(Sys.time(), "%F %R"),
    ": Creating Generator Seed",
    seed,
    "\n")
  preloadPLG2(
    gen_rds,
    1200,
    paste0(
      preloadGeneratorpath,
      "deepSea/batchsize32_maxlen1000_max_samples8_seed_",
      seed,
      ".rds"
    ),
    path, bs
  )
  cat(format(Sys.time(), "%F %R"), ": FINISHED\n")
}
