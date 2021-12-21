library(deepG)
host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- "/vol/projects/BIFO/genomenet/BacDive/datagram/"
  preloadGeneratorpath <- "/vol/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log      <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- "/net/projects/BIFO/genomenet/BacDive/datagram/"
  preloadGeneratorpath <- "/net/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log      <- "/net/projects/BIFO/genomenet/tensorboard/yin"
} else if (host %in% c("mcml-d", "gpu-00", "dgx-00", "datala")) {
  setwd("/home/ru39qiq5/MAGenomeNet")
  savepath             <- "/home/ru39qiq5/genedata/BacDive/datagram/"
  preloadGeneratorpath <- "/home/ru39qiq5/genedata/preprocessed_data/"
  tensorboard.log      <- "/home/ru39qiq5/tensorboard"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  savepath             <- "/genedata/genomes/BacDive/datagram/"
  preloadGeneratorpath <- "/genedata/genomes/preprocessed_data/"
  tensorboard.log      <- "/home/user/tensorboard"
}

path.val <- as.list(paste0(savepath, list("negative", "positive"), "/validation/"))

source("R/help/preloadedGenerator.R")
bs <- 32

for (ml in c(6700, 7000, 7600)) {
  for (seed in c(1234, 111, 222, 888, 999)) {
    cat(format(Sys.time(), "%F %R"),
      ": Creating Generator Seed",
      seed,
      "\n")
    initializeGenerators(
      path.val,
      batch.size = bs,
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
        preloadGeneratorpath, "gram/batchsize",
        bs,
        "_maxlen",
        ml,
        "_max_samples8_seed_",
        seed,
        ".rds"
      ),
      F, bs, NULL, bs, path.val, NULL, ml
    )
    cat(format(Sys.time(), "%F %R"), ": FINISHED\n")
  }
}