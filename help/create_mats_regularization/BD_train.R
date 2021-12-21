source("R/help/getenvBacDive.R")

print("CREATING MATRIX FOR TRAINING DATA")

asd <- fastaLabelGenerator(
  path,
  batch.size = 1,
  maxlen = 6700,
  step = 6700,
  file_filter = filestrain$V1,
  fileLog = paste0(sub("\\/data\\/.*", "", path), "/GRAM/fullgentrain.csv"),
  target_from_csv = cats
)

i <- 1
representations <- list()
label <- list()
file <- list()
rowsbefore <- 0
while (nrow(fread(paste0(
  sub("\\/data\\/.*", "", path), "/GRAM/fullgentrain.csv"
), header = F)) <= nrow(readRDS(paste0(
  sub("\\/data\\/.*", "", path), "/Gramtrain.rds"
)))) {
  dat <- asd()
  representations[[i]] <-
    as.array(dat$X %>% tf$convert_to_tensor(),
      nrow = 6700,
      ncol = 4)
  label[[i]] <- dat$Y[, 1]
  files <-
    fread(paste0(sub("\\/data\\/.*", "", path), "/GRAM/fullgentrain.csv"), header = F)
  rowsnow <- nrow(files)
  file[[i]] <- files[rowsnow, 1]
  i <- i + 1
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    if (rowsbefore %% 100 == 0) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
    }
  }
}

saveRDS(list(representations, label, file),
  paste0(sub("\\/data\\/.*", "", path), "/GRAM/fullmattrain.rds"))
