auc_roc_metric_ds <- function(file_path,
  model = NULL,
  batch_size = 100,
  numberOfBatches = 10,
  #roc_plot_path = NULL,
  return_auc_vector = FALSE,
  evaluate_all_files = FALSE,
  return_auprc_vector = FALSE,
  seed = 1234) {
  format <- "rds"
  plot_roc <- FALSE #!is.null(roc_plot_path)
  test_loss <- tf$keras$metrics$Mean(name = 'test_loss')
  
  if (evaluate_all_files) {
    rds_files <- list_fasta_files(corpus.dir = file_path,
      format = "rds",
      file_filter = NULL)
    num_samples <- 0
    for (file in rds_files) {
      rds_file <- readRDS(file)
      num_samples <- dim(rds_file[[1]])[1] + num_samples
    }
    numberOfBatches <- ceiling(num_samples / batch_size)
    message_string <-
      paste0(
        "Evaluate ",
        num_samples,
        " samples. Setting numberOfBatches to ",
        numberOfBatches,
        "."
      )
    message(message_string)
  }
  
  if (length(model$outputs) > 1) {
    stop("model should have only one output layer")
  }
  
  num_layers <- length(model$get_config()$layers)
  layer_name <- model$get_config()$layers[[num_layers]]$name
  activation_string <-
    as.character(model$get_layer(layer_name)$activation)
  if (!(stringr::str_detect(activation_string, "sigmoid"))) {
    stop("model should have sigmoid activation at last layer")
  }
  
  set.seed(seed)
  gen <-
    gen_rds(rds_folder = file_path,
      batch_size = batch_size,
      fileLog = NULL)
  y_conf_list <- vector("list", numberOfBatches)
  y_true_list <- vector("list", numberOfBatches)
  j <- 1
  for (batch in 1:numberOfBatches) {
    z <- gen()
    x <- z[[1]][,1:900,]
    y <- z[[2]]
    y_conf <- predict(model, x, verbose = 0)
    loss <- loss_binary_crossentropy(y, y_conf) %>%
      tf$stack(axis = 0) %>% tf$reduce_mean()
    test_loss(loss)
    
    if (batch_size == 1) {
      y_conf_list[[batch]] <-
        matrix(y_conf, nrow = 1) %>% as.data.frame()
      y_true_list[[batch]] <-
        matrix(y, nrow = 1) %>% as.data.frame()
    } else {
      y_conf_list[[batch]] <- y_conf %>% as.data.frame()
      y_true_list[[batch]] <- y  %>% as.data.frame()
    }
    if (batch %in% ceiling(seq(1, numberOfBatches, by = numberOfBatches / 10))) {
      cat(format(Sys.time(), "%F %R : Predictions"), j*10, "% done \n")
      j <- j + 1
    }
  }
  
  df_conf <- data.table::rbindlist(y_conf_list) %>% as.data.frame()
  df_true <- data.table::rbindlist(y_true_list) %>% as.data.frame()
  
  
  if (evaluate_all_files) {
    df_conf <- df_conf[1:num_samples,]
    df_true <- df_true[1:num_samples,]
  }
  
  zero_var_col <- apply(df_true, 2, stats::var) == 0
  if (sum(zero_var_col) > 0) {
    warning_message <-
      paste(
        sum(zero_var_col),
        "columns contain just one label and will be removed from evaluation"
      )
    df_conf <- df_conf[, !zero_var_col]
    df_true <- df_true[, !zero_var_col]
    warning(warning_message)
  }
  
  
  cat(format(Sys.time(), "%F %R :"), "F1 AND BALANCED ACCURACY CALCULATION\n")
  f1 <- list("f1_per_class1" = list(), "f1_per_class0" = list(), "balanced_f1" = list())
  balacc <- list()
  pred_matrix <- ifelse(df_conf < 0.5, 0, 1)
  j <- 1 
  for (i in 1:ncol(pred_matrix)) {
    cm <- table(factor(as.character(pred_matrix[, i]), levels = c("0", "1")), factor(as.character(df_true[, i]), levels = c("0", "1")))
    f1b <- f1_from_conf_matrix(cm)
    f1$f1_per_class1[[i]] <- f1b$f1_per_class[1]
    f1$f1_per_class0[[i]] <- f1b$f1_per_class[2]
    f1$balanced_f1[[i]] <- f1b$balanced_f1
    balacc[[i]] <- ((cm[2, 2] / (cm[2, 2] + cm[1, 2])) + (cm[1, 1] / (cm[1, 1] + cm[2, 1]))) / 2 #(((TP/(TP+FN)+(TN/(TN+FP))) / 2
    if (i %in% ceiling(seq(1, ncol(pred_matrix), by = ncol(pred_matrix) / 10))) {
      cat(format(Sys.time(), "%F %R :"), j*10, "% done \n")
      j <- j + 1
    }
  }
  
  ## auc with pROC package
  #auc_list <- purrr::map(1:ncol(df_conf), ~pROC::roc(df_true[ , .x], df_conf[ , .x], quiet = TRUE))
  
  cat(format(Sys.time(), "%F %R :"), "AUC AND AUPRC CALCULATION\n")
  # auc and auprc with PRROC package
  auc_list <- purrr::map(
    1:ncol(df_conf),
    ~ PRROC::roc.curve(scores.class0 = df_conf[, .x],
      weights.class0 = df_true[, .x])
  )
  auc_vector <- vector("numeric", ncol(df_true))
  for (i in 1:length(auc_vector)) {
    auc_vector[i] <- auc_list[[i]]$auc
  }
  
  auprc_list <- purrr::map(
    1:ncol(df_conf),
    ~ PRROC::pr.curve(scores.class0 = df_conf[, .x],
      weights.class0 = df_true[, .x])
  )
  auprc_vector <- vector("numeric", ncol(df_true))
  for (i in 1:length(auprc_vector)) {
    auprc_vector[i] <- auprc_list[[i]]$auc.integral
  }
  
  output_auc <-
    list(
      mean_auc = mean(auc_vector),
      median_auc = median(auc_vector),
      summary = summary(auc_vector),
      standard_deviation = sd(auc_vector)
    )
  
  if (return_auc_vector) {
    output_auc$auc_vector <- auc_vector
  }
  
  output_auprc <-
    list(
      mean_auprc = mean(auprc_vector),
      median_auprc = median(auprc_vector),
      summary = summary(auprc_vector),
      standard_deviation = sd(auprc_vector)
    )
  
  if (return_auprc_vector) {
    output_auprc$auprc_vector <- auprc_vector
  }
  
  output <- list(AUC = output_auc, AUPRC = output_auprc, 
    f1_0 = summary(unlist(f1$f1_per_class0)), f1_1 = summary(unlist(f1$f1_per_class1)), f1_bal = summary(unlist(f1$balanced_f1)), loss = as.double(test_loss$result()), bal_accuracy = summary(unlist(balacc)))
  
  if (plot_roc) {
    roc_plot <- pROC::ggroc(data = auc_list,
      linetype = 1,
      size = 0.2) +
      theme_minimal() + theme(legend.position = "none") +
      geom_segment(aes(
        x = 1,
        xend = 0,
        y = 0,
        yend = 1
      ),
        color = "blue",
        linetype = "dashed") +
      scale_colour_manual(values = rep("black", length(auc_list)))
    ggsave(plot = roc_plot, filename =  roc_plot_path)
  }
  
  return(output)
}

list_fasta_files <- function(corpus.dir, format, file_filter) {
  if (is.list(corpus.dir)) {
    fasta.files <- list()
    for (i in 1:length(corpus.dir)) {
      
      if (endsWith(corpus.dir[[i]], paste0(".", format))) {
        fasta.files[[i]] <- corpus.dir[[i]]
        
      } else {
        
        fasta.files[[i]] <- list.files(
          path = xfun::normalize_path(corpus.dir[[i]]),
          pattern = paste0("\\.", format, "$"),
          full.names = TRUE)
      }
    }
    fasta.files <- unlist(fasta.files)
    num_files <- length(fasta.files)
  } else {
    
    # single file
    if (endsWith(corpus.dir, paste0(".", format))) {
      num_files <- 1 
      fasta.files <- corpus.dir   
    } else {
      
      fasta.files <- list.files(
        path = xfun::normalize_path(corpus.dir),
        pattern = paste0("\\.", format, "$"),
        full.names = TRUE)
      num_files <- length(fasta.files)
    }
  }
  
  if (!is.null(file_filter)) {
    fasta.files <- fasta.files[basename(fasta.files) %in% file_filter]
    if (length(fasta.files) < 1) {
      stop_text <- paste0("None of the files from ", unlist(corpus.dir), 
        " are present in train_val_split_csv table for either train or validation. \n")
      stop(stop_text)
    }
  }
  
  return(fasta.files)
}
