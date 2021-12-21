library(deepG)
host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)
species <- list("human_0_1", "bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
lochelm <- "/projects/BIFO/genomenet/training_data/"
locelek <- "/genedata/genomes/"

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  path            <- as.list(paste0("/vol", lochelm, species, "/train/"))
  path.val        <- as.list(paste0("/vol", lochelm, species, "/validation/"))
  tensorboard.log <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
  
  cats            <- "/vol/projects/BIFO/genomenet/BacDive/GCA_gram_two_classes.csv"
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  path            <- as.list(paste0("/net", lochelm, species, "/train/"))
  path.val        <- as.list(paste0("/net", lochelm, species, "/validation/"))
  tensorboard.log <- "/net/projects/BIFO/genomenet/tensorboard/yin"
  
  cats            <- "/net/projects/BIFO/genomenet/BacDive/GCA_gram_two_classes.csv"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  path            <- as.list(paste0(locelek, species, "/train/"))
  path.val        <- as.list(paste0(locelek, species, "/validation/"))
  tensorboard.log <- "/home/user/tensorboard"
  
  cats            <- "GRAMtest.csv"
}
source("R/help/preloadedGenerator.R")
bs <- 32

for (ml in c(6700, 7000, 7600)) {
  for (seed in c(111, 1234, 222, 888, 999)) {
    cat(format(Sys.time(), "%F %R"),
      ": Creating Generator Seed",
      seed, "maxlen", ml, 
      "\n")
    preloadPLG2(
      fastaFileGenerator,
      1200,
      paste0(
        "/net/projects/BIFO/genomenet/preprocessed_data/all/batchsize",
        bs,
        "_maxlen",
        ml,
        "_max_samples8_seed_",
        seed,
        ".rds"
      ),
      path.val,
      batch.size = bs,
      maxlen = ml,
      step = ml,
      max_samples = 8,
      randomFiles = TRUE,
      seed = seed,
      sample_by_file_size = T
    )
    cat(format(Sys.time(), "%F %R"), ": FINISHED\n")
  }
}