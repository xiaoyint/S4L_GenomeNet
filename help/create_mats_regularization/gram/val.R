args = commandArgs(trailingOnly = TRUE)
spec      <- as.integer(args[1])
gramstain <- ifelse(spec == 1, "negative", "positive")
source("R/help/getenvBacDive.R")

suppressMessages(library(deepG))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))
suppressMessages(library(reticulate))
suppressMessages(library(data.table))
suppressMessages(library(glmnet))
suppressMessages(library(hdf5r))
library(stringi)

print("CREATING MATRIX FOR VALIDATION DATA")

fasval <- fastaFileGenerator(path.val[[spec]], batch.size = 1, maxlen = 6700, step = 6700, fileLog = paste0(savepath, gramstain, "/fullgenval.csv"))

i <- 1
representations <- list()
file <- list()
rowsbefore <- 0
num_files <- length(list.files(path.val[[spec]]))
cat(num_files, "in total\n")
rowsnow <- 0

while (rowsnow < num_files) {
  dat <- fasval()
  rep <- dat$X %>% tf$convert_to_tensor()
  
  files <- fread(paste0(savepath, gramstain, "/fullgenval.csv"), header = F)
  representations[[i]] <- dat$X
  file[[i]] <- tail(files,1)$V1
  
  rowsnow <- nrow(files)
  i <- i + 1
  
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    if (rowsbefore %% 100 == 0) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
    }
  }
}

saveRDS(list(representations, file), paste0(savepath, gramstain, "/fullmatval.rds"))
