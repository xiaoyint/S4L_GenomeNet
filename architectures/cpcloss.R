cpc <- function(latents,
  context,
  target_dim = 64,
  emb_scale = 0.1 ,
  steps_to_ignore = 2,
  steps_to_predict = 3,
  steps_skip = 1,
  batch.size = 32,
  k = 5,
  ACGT = FALSE,
  onlysameposition = FALSE) {
  # define empty lists for metrics
  #a_complement <- tf$convert_to_tensor(array(as.array(a)[ , (dim(a)[2]):1, 4:1], dim = c(dim(a)[1],dim(a)[2],dim(a)[3])))
  loss <- list()
  acc <- list()
  
  # create context tensor
  ctx <- context(latents)
  
  c_dim <- latents$shape[[2]]
  
  
  # loop for different distances of predicted patches
  for (i in seq(steps_to_ignore, (steps_to_predict - 1), steps_skip)) {
    # define patches to be deleted
    c_dim_i <- c_dim - i - 1
    
    # add anils architecture
    if (isTRUE(ACGT)) {
      if (isTRUE(onlysameposition)) {
        # define total number of elements in context tensor
        total_elements <- batch.size * c_dim
        # total number of elements is batch size, as only one position per batch is taken
        # add conv layer and reshape for matrix multiplication
        both <-
          ctx %>% layer_conv_1d(kernel_size = 1,
            filters = target_dim)
        
        preds_i <- both[1:batch.size, , ]
        revcompl <-
          both[(batch.size + 1):as.integer(batch.size * 2), ,]
        
        for (j in seq_len(dim(both)[[2]] - (i + 1))) {
          preds_ij <- preds_i[, j, ] %>%
            k_reshape(c(-1, target_dim)) * emb_scale
          
          revcompl_j <-
            revcompl[, (dim(both)[[2]] - (j - 1) - (i + 1)),] %>%
            k_reshape(c(-1, target_dim))
          
          logits <- tf$matmul(preds_ij, tf$transpose(revcompl_j))
          
          # always the patch from the same batch is the true one
          labels <- floor(seq(batch.size - 1 , 0)) %>% as.integer()
          
          # calculate loss and accuracy for each step
          loss[[length(loss) + 1]] <-
            tf$nn$sparse_softmax_cross_entropy_with_logits(labels, logits) %>%
            tf$stack(axis = 0) %>% tf$reduce_mean()
          acc[[length(acc) + 1]] <-
            tf$keras$metrics$sparse_top_k_categorical_accuracy(tf$cast(labels, dtype = "int64"),
              logits,
              as.integer(k)) %>%
            tf$stack(axis = 0) %>% tf$reduce_mean()
        }
        
      } else {
        # define total number of elements in context tensor
        total_elements <- batch.size * c_dim
        # add conv layer and reshape for matrix multiplication
        targets <-
          ctx %>% layer_conv_1d(kernel_size = 1,
            filters = target_dim)
        preds_i <- targets[1:batch.size, , ]
        preds_i <- k_reshape(preds_i, c(-1, target_dim)) * emb_scale
        
        # add possibility to predict context patches instead of latents tensor
        # split original and reverse complement sequences for prediction
        revcompl <-
          targets[(batch.size + 1):as.integer(batch.size * 2), , ]
        
        targets2 <- k_reshape(revcompl, c(-1, target_dim))
        logits <- tf$matmul(preds_i, tf$transpose(targets2))
        
        # get position of labels
        b <- floor(seq(total_elements - 1, 0) / c_dim)
        col <- seq(total_elements - 1, 0) %% c_dim
        
        # define labels
        labels <- b * c_dim + col - (i + 1)
        labels <- as.integer(labels)
      }
      
    } else {
      # define total number of elements in context tensor
      total_elements <- batch.size * c_dim_i
      # add conv layer and reshape tensor for matrix multiplication
      targets <-
        latents %>% layer_conv_1d(kernel_size = 1, filters = target_dim) %>% k_reshape(c(-1, target_dim))
      # add conv layer and reshape for matrix multiplication
      preds_i <-
        ctx %>% layer_conv_1d(kernel_size = 1,
          filters = target_dim)
      preds_i <- preds_i[, (1:(c_dim - i - 1)), ]
      preds_i <- k_reshape(preds_i, c(-1, target_dim)) * emb_scale
      
      # define logits normally
      logits <- tf$matmul(preds_i, tf$transpose(targets))
      
      # get position of labels
      b <- floor(seq(0, total_elements - 1) / c_dim_i)
      col <- seq(0, total_elements - 1) %% c_dim_i
      
      # define labels
      labels <- b * c_dim + col + (i + 1)
      labels <- as.integer(labels)
    }
    
    
    if (!isTRUE(onlysameposition)) {
      # calculate loss and accuracy for each step
      loss[[length(loss) + 1]] <-
        tf$nn$sparse_softmax_cross_entropy_with_logits(labels, logits) %>%
        tf$stack(axis = 0) %>% tf$reduce_mean()
      acc[[length(acc) + 1]] <-
        tf$keras$metrics$sparse_top_k_categorical_accuracy(tf$cast(labels, dtype = "int64"), logits, as.integer(k)) %>%
        tf$stack(axis = 0) %>% tf$reduce_mean()
    }
  }
  
  # convert to tensor for output
  loss <-
    loss %>% tf$stack(axis = 0) %>% tf$reduce_mean()
  acc <- acc %>% tf$stack(axis = 0) %>% tf$reduce_mean()
  return(tf$stack(list(loss, acc)))
}
