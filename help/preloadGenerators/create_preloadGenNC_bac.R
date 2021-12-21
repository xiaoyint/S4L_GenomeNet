library(deepG)
host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)
species <- list("bacteria_1_3")
lochelm <- "/projects/BIFO/genomenet/training_data/"
locelek <- "/genedata/genomes/"

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  path.val        <- as.list(paste0("/vol", lochelm, species, "/validation/"))
  preloadGeneratorpath <- "/vol/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  path.val        <- as.list(paste0("/net", lochelm, species, "/validation/"))
  preloadGeneratorpath <- "/net/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log <- "/net/projects/BIFO/genomenet/tensorboard/yin"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  path.val        <- as.list(paste0(locelek, species, "/validation/"))
  preloadGeneratorpath <- "/genedata/genomes/preprocessed_data/"
  tensorboard.log <- "/home/user/tensorboard"
}
source("R/help/preloadedGenerator.R")

for (seed in c(1234, 111, 222, 888, 999)) {
  for (ml in c(500, 1000, 2000)) {
    cat(format(Sys.time(), "%F %R"),
      ": Creating Generator Seed",
      seed, "maxlen", ml, 
      "\n")
    preloadPLG2(
      fastaFileGenerator,
      1200,
      # = 3000*0.4
      paste0(
        preloadGeneratorpath, "bacteria/batchsize",
        512,
        "_maxlen",
        ml,
        "_max_samples8_seed_",
        seed,
        ".rds"
      ),
      path.val,
      batch.size = 512,
      maxlen = ml,
      step = ml,
      max_samples = 8,
      randomFiles = TRUE,
      seed = seed,
      sample_by_file_size = T
    )
    cat(format(Sys.time(), "%F %R"), ": FINISHED", seed, "\n")
  }
}