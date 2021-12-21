source("R/help/getenvdeepSea.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "LOAD TRAIN DATA\n")

for (j in c(0.1, 1, 10)) {
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "LOAD" , j, "% TRAIN DATA\n")
  gentr <-
    gen_rds(paste0(paste0(strsplit(path, "/")[[1]][1:7], collapse = "/"), "/subset_", j, "_perc.rds"), 1, paste0(path, "/fullgentrain", j, ".csv"))
  
  sequemce <- list()
  label <- list()
  
  rds_file <-
    readRDS(paste0(
      paste0(strsplit(path, "/")[[1]][1:7], collapse = "/"),
      "/subset_", j, "_perc.rds"))
  num_samples <- dim(rds_file[[1]])[1]
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "CREATING MATRIX WITH " , num_samples, " ROWS\n")
  
  for (i in seq(num_samples)) {
    dat <- gentr()
    sequemce[[i]] <- as.array(dat[[1]] %>% tf$convert_to_tensor(),
      nrow = 1000,
      ncol = 4)
    label[[i]] <- dat[[2]]
    if (length(sequemce) %in% ceiling(seq(0, num_samples, num_samples / 10))) {
      cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), " next 10% done\n")
    }
  }
  
  saveRDS(list(sequemce, label), paste0(path, "/fullmattrain", j, ".rds"))
  cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), j, "% TRAIN MATRIX SAVED\n")
}
