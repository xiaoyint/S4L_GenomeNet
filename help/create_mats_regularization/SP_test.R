species <- list("human_0_1", "bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
source("R/help/getenv.R")
suppressMessages(library(deepG))
suppressMessages(library(tensorflow))
suppressMessages(library(keras))
suppressMessages(library(reticulate))
suppressMessages(library(data.table))
suppressMessages(library(glmnet))
suppressMessages(library(hdf5r))
library(stringi)

print("CREATING MATRIX FOR TEST DATA")

initializeGenerators(
  path.tes,
  batch.size = 4,
  maxlen = 6700,
  step = 6700,
  fileLog = paste0(savepath, "fullgentest.csv")
)
fastest <- labelByFolderGeneratorWrapper(F, 4, NULL, 4, path.tes, NULL, 6700)

i <- 1
representations <- list()
label <- list()
file <- list()
rowsbefore <- 0
num_files <- 0
for (j in seq_along(path.tes)) {
  num_files <- num_files + length(list.files(path.tes[[j]]))
}
cat(num_files, "in total\n")
rowsnow <- 0
while (rowsnow < num_files) {
  dat <- fastest()
  rep <- dat$X %>% tf$convert_to_tensor()
  
  files <- fread(paste0(savepath, "fullgentest.csv"), header = F)
  ar <- list()
  fidt <- list()
  spelist <- c("human", "bacteria", "viral-phage", "viral-no-phage")
  for (k in 1:4) {
    ar[[k]] <- as.array(rep[k,1:6700,1:4], nrow = 6700, ncol = 4)
    representations[[i + k - 1]]   <- ar[[k]]
    label[[i + k - 1]] <- data.table(k)
    
    fidt[[k]] <- data.table(stri_locate_last(files$V1, fixed = spelist[[k]]))
    fidt[[k]]$row <- rownames(fidt[[k]])
    file[[i + k - 1]] <- files[as.integer(max(fidt[[k]][!is.na(start),row])), 1]
  }
  rowsnow <- nrow(files)
  i <- i + 4
  
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    if (rowsbefore %% 100 == 0) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
    }
  }
}

saveRDS(list(representations, label, file), paste0(savepath, "fullmattest.rds"))
