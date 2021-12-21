source("R/help/getenvdeepSea.R")
source("R/help/auc.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "LOAD TRAIN DATA\n")

gentr <- gen_rds(paste0(path, "/train"), 1, paste0(path, "/fullgentrain.csv"))

i <- 1
sequemce <- list()
label <- list()
file <- list()

rds_files <- list_fasta_files(corpus.dir = paste0(path, "/train"),
  format = "rds",
  file_filter = NULL)
num_samples <- 0
for (file in rds_files) {
  rds_file <- readRDS(file)
  num_samples <- dim(rds_file[[1]])[1] + num_samples
}
cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "CREATING MATRIX WITH " , num_samples, " ROWS\n")

for (i in seq(num_samples)) {
  dat <- gentr()
  sequemce[[i]] <- as.array(dat[[1]] %>% tf$convert_to_tensor(),
      nrow = 1000,
      ncol = 4)
  label[[i]] <- dat[[2]]
  files <- fread(paste0(path, "/fullgentrain.csv"), header = F)
  rowsnow <- nrow(files)
  file[[i]] <- files[rowsnow, 1]
  if (length(sequemce) %in% ceiling(seq(0,num_samples,num_samples/10))) {
    cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " next 10% done\n")
  }
}

saveRDS(list(sequemce, label, file),
  paste0(path, "/fullmattrain.rds"))
