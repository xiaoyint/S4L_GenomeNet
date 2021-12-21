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

if (host == "bioinf") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- "/vol/projects/BIFO/genomenet/BacDive/datagram/"
  preloadGeneratorpath <- "/vol/projects/BIFO/genomenet/preprocessed_data/"
  tensorboard.log      <- "/vol/projects/BIFO/genomenet/tensorboard/yin"
  tf$config$threading$set_intra_op_parallelism_threads(1L)  
  tf$config$threading$set_inter_op_parallelism_threads(1L)
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
} else if (host == "pmuenc") {
  setwd("/home/xto/MAGenomeNet")
  savepath             <- "/home/xto/genedata/BacDive/datagram/"
  preloadGeneratorpath <- "/home/xto/genedata/preprocessed_data/"
  tensorboard.log      <- "/home/xto/tensorboard"
}

path            <- as.list(paste0(savepath, list("negative", "positive"), "/train/"))
path.val        <- as.list(paste0(savepath, list("negative", "positive"), "/validation/"))
path.tes        <- as.list(paste0(savepath, list("negative", "positive"), "/test/"))

gpus = tf$config$experimental$list_physical_devices('GPU')
lapply(gpus, function(g)
  tf$config$experimental$set_memory_growth(g, TRUE))

`%ni%` <- Negate("%in%")
rm(host)