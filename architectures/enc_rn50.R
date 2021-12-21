convblock <- function(X, f, stage, block, s = 2) {
  X_shortcut <- X
  
  cname <- paste("conv", stage, "_block", block, sep = "")
  ks <- list(1, 3, 1)
  pad <- list("VALID", "SAME", "VALID")
  s <- list(s, 1, 1)
  
  for (i in seq(3)) {
    X <- X %>%
      layer_conv_1d(
        f[[i]],
        kernel_size = ks[[i]],
        strides = s[[i]],
        padding = pad[[i]],
        name = paste(cname, "_", i, "_conv", sep = "")
      ) %>%
      layer_batch_normalization(axis = 2,
        name = paste(cname, "_", i, "_bn", sep = ""))
    if (i < 3) {
      X <- X %>%
        layer_activation_relu(name = paste(cname, "_", i, "_relu", sep = ""))
    }
    
  }
  
  X_shortcut <- X_shortcut %>% layer_conv_1d(
    filters = f[[3]],
    kernel_size = 1,
    strides = s[[1]],
    padding = "VALID",
    name = paste(cname, "_sc_conv", sep = "")
  ) %>%
    layer_batch_normalization(axis = 2,
      name = paste(cname, "_sc_bn", sep = ""))
  
  X <-
    layer_add(c(X, X_shortcut), name = paste(cname, "_add", sep = "")) %>%
    layer_activation_relu(name = paste(cname, "_out", sep = ""))
}


idenblock <- function(X, f, stage, block) {
  X_shortcut <- X
  
  cname <- paste("conv", stage, "_block", block, sep = "")
  ks <- list(1, 3, 1)
  pad <- list("VALID", "SAME", "VALID")
  
  
  for (i in seq(3)) {
    X <- X %>%
      layer_conv_1d(
        f[[i]],
        kernel_size = ks[[i]],
        strides = 1,
        padding = pad[[i]],
        name = paste(cname, "_", i, "_conv", sep = "")
      ) %>%
      layer_batch_normalization(axis = 2,
        name = paste(cname, "_", i, "_bn", sep = ""))
    if (i < 3) {
      X <- X %>%
        layer_activation_relu(name = paste(cname, "_", i, "_relu", sep = ""))
    }
    
  }
  
  X <-
    layer_add(c(X, X_shortcut), name = paste(cname, "_add", sep = "")) %>%
    layer_activation_relu(name = paste(cname, "_out", sep = ""))
}

encoder <-
  function(maxlen = NULL,
    patchlen =  NULL,
    nopatches = NULL,
    lesschannels = FALSE,
    threeblocks = FALSE,
    eval = FALSE) {
    
    if (is.null(nopatches)) {
      source("R/help/calc_help.R")
      nopatches <- nopatchescalc(patchlen, maxlen, patchlen * 0.4)
    }
    
    if (eval) {
      resinput <- layer_input(shape = c(maxlen, 4))
      resreshape <- resinput
    } else {
      resinput <- layer_input(shape = c(maxlen, 4))
      stridelen <- as.integer(0.4 * patchlen)
      resreshape <- resinput %>%
        layer_reshape(list(maxlen, 4L, 1L), name = "prep_reshape1", dtype = "float32") %>%
        tf$image$extract_patches(
          sizes = list(1L, patchlen, 4L, 1L),
          strides = list(1L, stridelen, 4L, 1L),
          rates = list(1L, 1L, 1L, 1L),
          padding = "VALID",
          name = "prep_patches"
        ) %>%
        layer_reshape(list(nopatches, patchlen, 4L),
          name = "prep_reshape2") %>%
        tf$reshape(list(-1L, patchlen, 4L),
          name = "prep_reshape3")
      # zero padding?
    }
    
    if (isTRUE(lesschannels)) {
      size <- c(32, 64, 128, 256, 512)
    }
    else{
      size <- c(64, 128, 256, 512, 1024)
    }
    
    # Stage 1
    stage1 <- resreshape %>%
      layer_conv_1d(
        filters = size[1],
        kernel_size = 5,
        strides = 2,
        name = "conv1_conv"
      ) %>%
      layer_batch_normalization(axis = 2,
        name = "conv1_bn") %>%
      layer_activation_relu(name = "conv1_relu") %>%
      layer_max_pooling_1d(pool_size = 2L,
        strides = 2,
        name = "conv1_pool")
    
    # Stage 2
    stage2 <-
      convblock(
        stage1,
        list(size[1], size[1], size[3]),
        stage = 2,
        block = 1,
        s = 1
      ) %>%
      idenblock(list(size[1], size[1], size[3]),
        stage = 2,
        block = 2) %>%
      idenblock(list(size[1], size[1], size[3]),
        stage = 2,
        block = 3)
    
    if (threeblocks) {
      stage3 <-
        convblock(
          stage2,
          list(size[2], size[2], 2048),
          stage = 3,
          block = 1,
          s = 2
        ) %>%
        idenblock(list(size[2], size[2], 2048),
          stage = 3,
          block = 2) %>%
        idenblock(list(size[2], size[2], 2048),
          stage = 3,
          block = 3) %>%
        idenblock(list(size[2], size[2], 2048),
          stage = 3,
          block = 4)
      
      final <-
        stage3 %>% layer_global_average_pooling_1d() %>%
        layer_flatten() %>%
        tf$reshape(list(-1L, tf$cast(nopatches, tf$int16), 2048L))
      
    } else {
      # Stage 3
      stage3 <-
        convblock(
          stage2,
          list(size[2], size[2], size[4]),
          stage = 3,
          block = 1,
          s = 2
        ) %>%
        idenblock(list(size[2], size[2], size[4]),
          stage = 3,
          block = 2) %>%
        idenblock(list(size[2], size[2], size[4]),
          stage = 3,
          block = 3) %>%
        idenblock(list(size[2], size[2], size[4]),
          stage = 3,
          block = 4)
      
      # Stage 4
      stage4 <-
        convblock(
          stage3,
          list(size[3], size[3], size[5]),
          stage = 4,
          block = 1,
          s = 2
        ) %>%
        idenblock(list(size[3], size[3], size[5]),
          stage = 4,
          block = 2) %>%
        idenblock(list(size[3], size[3], size[5]),
          stage = 4,
          block = 3) %>%
        idenblock(list(size[3], size[3], size[5]),
          stage = 4,
          block = 4) %>%
        idenblock(list(size[3], size[3], size[5]),
          stage = 4,
          block = 5) %>%
        idenblock(list(size[3], size[3], size[5]),
          stage = 4,
          block = 6)
      
      # Stage 5
      stage5 <-
        convblock(
          stage4,
          list(size[4], size[4], 2048),
          stage = 5,
          block = 1,
          s = 2
        ) %>%
        idenblock(list(size[4], size[4], 2048),
          stage = 5,
          block = 2) %>%
        idenblock(list(size[4], size[4], 2048),
          stage = 5,
          block = 3)
      
      final <-
        stage5 %>% layer_global_average_pooling_1d() %>%
        layer_flatten() %>%
        tf$reshape(list(-1L, tf$cast(nopatches, tf$int16), 2048L))
    }
    keras_model(resinput, final)
    #final2 <- final %>% layer_dense(2048, activation = "softmax")
  }
