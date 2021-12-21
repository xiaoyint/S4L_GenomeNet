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

print("CREATING MATRIX FOR TRAINING DATA")

initializeGenerators(
  path,
  batch.size = 4,
  maxlen = 6700,
  step = 6700,
  fileLog = paste0(savepath, "fullgentrain.csv")
)
fastrain <- labelByFolderGeneratorWrapper(F, 4, NULL, 4, path, NULL, 6700)

i <- 1
representations <- list()
label <- list()
file <- list()
rowsbefore <- 0
num_files <- 0
for (j in seq_along(path)) {
  num_files <- num_files + length(list.files(path[[j]]))
}
cat(num_files, "in total\n")
rowsnow <- 0
while (rowsnow < num_files) {
  dat <- fastrain()
  rep <- dat$X %>% tf$convert_to_tensor()
  
  files <- fread(paste0(savepath, "fullgentrain.csv"), header = F)
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
  # human <- as.array(rep[1,1:6700,1:4], nrow = 6700, ncol = 4)
  # bac   <- as.array(rep[2,1:6700,1:4], nrow = 6700, ncol = 4)
  # virp  <- as.array(rep[3,1:6700,1:4], nrow = 6700, ncol = 4)
  # virn  <- as.array(rep[4,1:6700,1:4], nrow = 6700, ncol = 4)
  #  
  # representations[[i]]   <- human
  # label[[i]] <- data.table(0)
  # representations[[i + 1]] <- bac
  # label[[i + 1]] <- data.table(1)
  # representations[[i + 2]] <- virp
  # label[[i + 2]] <- data.table(2)
  # representations[[i + 3]] <- virn
  # label[[i + 3]] <- data.table(3)
  # 
  # files <- fread(paste0(savepath, "fullgentrain.csv"), header = F)
  # h <- data.table(stri_locate_last(files$V1, fixed = "human"))
  # h$row <- rownames(h)
  # b <- data.table(stri_locate_last(files$V1, fixed = "bacteria"))
  # b$row <- rownames(b)
  # vp <- data.table(stri_locate_last(files$V1, fixed = "viral-phage"))
  # vp$row <- rownames(vp)
  # vn <- data.table(stri_locate_last(files$V1, fixed = "viral-no-phage"))
  # vn$row <- rownames(vn)
  # 
  # file[[i]] <- files[as.integer(max(h[!is.na(start),row])), 1]
  # file[[i + 1]] <- files[as.integer(max(b[!is.na(start),row])), 1]
  # file[[i + 2]] <- files[as.integer(max(vp[!is.na(start),row])), 1]
  # file[[i + 3]] <- files[as.integer(max(vn[!is.na(start),row])), 1]
  rowsnow <- nrow(files)
  i <- i + 4
  
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    if (rowsbefore %% 100 == 0) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
    }
  }
}

saveRDS(list(representations, label, file), paste0(savepath, "fullmattrain.rds"))
