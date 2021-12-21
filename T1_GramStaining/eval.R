########################################################################################################
############################################## PREPARATION #############################################
########################################################################################################

cat(format(Sys.time(), "%F %R"), ": PREPARE VARIABLES\n")

args = commandArgs(trailingOnly = TRUE)
mod_name <- args[1] # "bl_lm_1"
source("R/help/getenvBacDive.R")

cat(format(Sys.time(), "%F %R :"), "READING MODEL\n")
modl <-
  keras::load_model_hdf5(
    paste0(
      "model_results/models/final/gram/",
      mod_name,
      "/bestmod.h5",
      sep = ""
    ),
    compile = F
  )


########################################################################################################
############################################## EVALUATION ##############################################
########################################################################################################

cat(format(Sys.time(), "%F %R :"), "START EVALUATION\n")
x <- evaluateFasta(
  fasta.path = path.tes,
  model = modl,
  batch.size = 400,
  step = 6700,
  filePath = paste0("model_results/models/final/gram/", mod_name),
  filename = "eval",
  mode = "label_folder",
  label_vocabulary = c("negative", "positive"),
  evaluate_all_files = T,
  verbose = TRUE,
  auc = T
)

cm <- x$confusion_matrix
balacc <-
  ((cm[2, 2] / (cm[2, 2] + cm[1, 2])) + (cm[1, 1] / (cm[1, 1] + cm[2, 1]))) / 2

results <- list(
  "Model" = mod_name,
  "TestMeasures" = x,
  "TestF1" = f1_from_conf_matrix(cm),
  "TestBalancedAccuracy" = balacc
)

#(((TP/(TP+FN)+(TN/(TN+FP))) / 2

results

sink(
  paste0(
    "model_results/models/final/gram/",
    mod_name,
    "/final_results",
    format(Sys.time(), "%F"),
    ".txt"
  )
)
print(results)
sink()
