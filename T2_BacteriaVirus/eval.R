########################################################################################################
############################################## PREPARATION #############################################
########################################################################################################

cat(format(Sys.time(), "%F %R"), ": PREPARE VARIABLES\n")

args = commandArgs(trailingOnly = TRUE)
mod_name <- args[1] # "bl_lm_1"

species <-
  list("bacteria_1_3", "viral-phage_1_3", "viral-no-phage_1_3")
source("R/help/getenv.R")

cat(format(Sys.time(), "%F %R :"), "READING MODEL\n")
modl <-
  keras::load_model_hdf5(
    paste0(
      "model_results/models/final/BacVir/",
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
  filePath = paste0("model_results/models/final/BacVir/", mod_name),
  filename = "eval",
  mode = "label_folder",
  label_vocabulary = c("bacteria", "viral-phage", "viral-no-phage"),
  evaluate_all_files = T,
  verbose = TRUE,
  auc = F
)

cm <- x$confusion_matrix

ccm <- list(
  "bac" = list(
    "TP" = cm[1, 1],
    "FP" = sum(cm[2:3, 1]),
    "TN" = sum(cm[2:3, 2:3]),
    "FN" = sum(cm[1, 2:3])
  ),
  "virph" = list(
    "TP" = cm[2, 2],
    "FP" = sum(cm[c(1, 3), 2]),
    "TN" = sum(cm[c(1, 3), c(1, 3)]),
    "FN" = sum(cm[2, c(1, 3)])
  ),
  "virnoph" = list(
    "TP" = cm[3, 3],
    "FP" = sum(cm[1:2, 3]),
    "TN" = sum(cm[1:2, 1:2]),
    "FN" = sum(cm[3, 1:2])
  )
)
Recall <- list(
  "bac" = ccm$bac$TP / (ccm$bac$TP + ccm$bac$FN),
  "virph" = ccm$virph$TP / (ccm$virph$TP + ccm$virph$FN),
  "virnoph" = ccm$virnoph$TP / (ccm$virnoph$TP + ccm$virnoph$FN)
)
Recall$bac <- ifelse(is.na(Recall$bac), 0, Recall$bac)
Recall$virph <- ifelse(is.na(Recall$virph), 0, Recall$virph)
Recall$virnoph <- ifelse(is.na(Recall$virnoph), 0, Recall$virnoph)
balacc <- (Recall$bac + Recall$virph + Recall$virnoph) / 3
results <- list(
  "Model" = mod_name,
  "TestMeasures" = x,
  "TestF1" = f1_from_conf_matrix(cm),
  "TestBalancedAccuracy" = balacc
)

results

sink(
  paste0(
    "model_results/models/final/BacVir/",
    mod_name,
    "/final_results",
    format(Sys.time(), "%F"),
    ".txt"
  )
)
print(results)
sink()
