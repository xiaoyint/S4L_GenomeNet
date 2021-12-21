suppressMessages(library(abind))
suppressMessages(library(data.table))
suppressMessages(library(deepG))
suppressMessages(library(dplyr))
suppressMessages(library(glmnet))
suppressMessages(library(hdf5r))
suppressMessages(library(keras))
suppressMessages(library(reticulate))
suppressMessages(library(stringr))
suppressMessages(library(tensorflow))

host <- stringr::str_sub(Sys.info()["nodename"], 1, 6)
lochelm <- "/projects/BIFO/genomenet/training_data/"

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- paste0("/vol", lochelm)
  tensorboard.log      <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
  preloadGeneratorpath <- "/vol/projects/BIFO/genomenet/preprocessed_data/"
  tf$config$threading$set_intra_op_parallelism_threads(1L)  
  tf$config$threading$set_inter_op_parallelism_threads(1L)
} else if (host == "luna-g") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- paste0("/net", lochelm)
  tensorboard.log      <- "/net/projects/BIFO/genomenet/tensorboard/yin"
  preloadGeneratorpath <- "/net/projects/BIFO/genomenet/preprocessed_data/"
} else if (host == "mcml-d") {
  setwd("/home/ru39qiq5/MAGenomeNet")
  savepath             <- "/home/ru39qiq5/genedata/"
  tensorboard.log      <- "/home/ru39qiq5/tensorboard"
  preloadGeneratorpath <- "/home/ru39qiq5/genedata/preprocessed_data/"
} else if (host == "gpuser") {
  setwd("/home/user/MAGenomeNet")
  savepath             <- "/genedata/genomes/"
  tensorboard.log      <- "/home/user/tensorboard"
  preloadGeneratorpath <- "/genedata/genomes/preprocessed_data/"
} else if (host == "pmuenc") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- "/home/xto/genedata/"
  tensorboard.log      <- "/home/xto/tensorboard"
  preloadGeneratorpath <- "/home/xto/genedata/preprocessed_data/"
}

path            <- as.list(paste0(savepath, species, "/train/"))
path.val        <- as.list(paste0(savepath, species, "/validation/"))
path.tes        <- as.list(paste0(savepath, species, "/test/"))

gpus = tf$config$experimental$list_physical_devices('GPU')
lapply(gpus, function(g)
  tf$config$experimental$set_memory_growth(g, TRUE))

rm(host, lochelm)
