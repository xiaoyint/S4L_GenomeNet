source("R/help/getenvBacDive.R")
library(stringi)

print("CREATING MATRIX FOR VALIDATION DATA")

initializeGenerators(
  path.val,
  batch.size = 2,
  maxlen = 6700,
  step = 6700,
  fileLog = paste0(sub("\\/negative\\/.*", "", path.val)[[1]], "/fullgenval.csv"),
)

fasval <-
  labelByFolderGeneratorWrapper(F, 1, NULL, 1, path.val, NULL, 6700)
num_files <- 0
for (i in seq_along(path.val)) {
  num_files <- num_files + length(list.files(path.val[[i]]))
}
cat(num_files, "in total\n")

i <- 1
representations <- list()
label <- list()
file <- list()
rowsbefore <- 0
rowsnow <- 0
while (rowsnow < num_files) {
  dat <- fasval()
  rep <- dat$X %>% tf$convert_to_tensor()
  
  files <- fread(paste0(sub("\\/negative\\/.*", "", path.val)[[1]], "/fullgenval.csv"), header = F)
  ar <- list()
  fidt <- list()
  lablist <- c("negative", "positive")
  for (k in 1:2) {
    ar[[k]] <- as.array(rep[k,1:6700,1:4], nrow = 6700, ncol = 4)
    representations[[i + k - 1]]   <- ar[[k]]
    label[[i + k - 1]] <- data.table(k)
    
    fidt[[k]] <- data.table(stri_locate_last(files$V1, fixed = lablist[[k]]))
    fidt[[k]]$row <- rownames(fidt[[k]])
    file[[i + k - 1]] <- files[as.integer(max(fidt[[k]][!is.na(start),row])), 1]
  }
  rowsnow <- nrow(files)
  i <- i + 2
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    if (rowsbefore %% 100 == 0) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
    }
  }
}

saveRDS(list(representations, label, file), paste0(sub("\\/negative\\/.*", "", path.val)[[1]], "/fullmatval.rds"))
