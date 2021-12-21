########################################################################################################
############################################## PREPARATION #############################################
########################################################################################################

cat(format(Sys.time(), "%F %R"), ": PREPARE VARIABLES\n")

args = commandArgs(trailingOnly = TRUE)
mod_name <- args[1] # "semi_rn18_fr_1
library(magrittr)

source("R/help/auc.R")
source("R/help/getenvdeepSea.R")

cat(format(Sys.time(), "%F %R :"), "READING MODEL\n")
modl <-
  keras::load_model_hdf5(
    paste0(
      "model_results/models/final/chrom/",
      mod_name,
      "/bestmod.h5",
      sep = ""
    ),
    compile = F
  )


########################################################################################################
############################################## EVALUATION ##############################################
########################################################################################################

cat(format(Sys.time(), "%F %R :"), "START AUC EVALUATION\n")
w <-
  auc_roc_metric_ds(
    file_path = paste0(path, "test"),
    model = modl,
    batch_size = 2048,
    return_auc_vector = T,
    return_auprc_vector = T,
    evaluate_all_files = T,
    seed = 1234
  )
cat(format(Sys.time(), "%F %R :"), "AUC EVALUATION FINISHED\n")

saveRDS(w, paste0("model_results/models/final/chrom/", mod_name, "/evaltest_", format(Sys.time(), "%F"), ".rds"))

cat(format(Sys.time(), "%F %R :"), "AUC EVALUATION SAVED\n")

results <- list(
  "Model" = mod_name,
  "TestAUC" = w[[1]][-5],
  "TestF1_class_0" = w[[3]],
  "TestF1_class_1" = w[[4]],
  "TestF1_balanced" = w[[5]],
  "TestLoss" = w[[6]],
  "TestBalancedAccuracy" = w[[7]]
)

results

sink(
  paste0(
    "model_results/models/final/chrom/",
    mod_name,
    "/evaltest_",
    format(Sys.time(), "%F"),
    ".txt"
  )
)
print(results)
sink()