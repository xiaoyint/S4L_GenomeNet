source("R/help/getenvdeepSea.R")

cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S :"), "LOAD VAL DATA\n")

genva <- gen_rds(pathval, 1,  paste0(pathval, "/fullgenval.csv"))

i <- 1
sequemce <- list()
label <- list()
file <- list()
rowsbefore <- 0
while (nrow(fread(paste0(pathval, "/fullgenval.csv"), header = F)) <= length(list.files(pathval))) {
  dat <- genva()
  sequemce[[i]] <-
    as.array(dat[[1]] %>% tf$convert_to_tensor(),
      nrow = 1000,
      ncol = 4)
  label[[i]] <- dat[[2]][, 1]
  i <- i + 1
  files <-
    fread(paste0(pathval, "/fullgenval.csv"), header = F)
  rowsnow <- nrow(files)
  file[[i]] <- files[rowsnow, 1]
  if (rowsnow != rowsbefore) {
    rowsbefore <- rowsnow
    cat(format(Sys.time(), "%d.%m.%y, %H:%M:%S"), rowsnow, " done\n")
  }
}

saveRDS(list(sequemce, label, file),
  paste0(pathval, "/fullmatval.rds"))
