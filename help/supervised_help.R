create_sup_model <-
  function(maxlen = NULL,
    patchlen = NULL,
    # one of "rn50", "rn18", "danQ":
    enc = "rn50") {
    source(paste0("R/architectures/enc_", enc, ".R"), local = T)
    source(paste0("R/architectures/ctx.R"), local = T)
    if (enc %in% c("rn50", "rn18")) {
      outlayer <- ifelse(enc == "rn50", 181, 77)
    } else if (enc == "danQ") {
      outlayer <- 14
    }
    latents <- encoder(maxlen = maxlen, patchlen =  patchlen)
    model <- keras_model(latents$input, context(latents$output))
    model <- keras_model(model$input,
      model$get_layer(model$layers[[outlayer]]$name)$get_output_at(0L))
    flat <- model$output %>% layer_global_average_pooling_1d()
    model <- keras_model(model$input, flat)
    return(model)
  }

reduce_model <-
  function(trainedmodel = NULL,
    maxlen = NULL,
    patchlen = NULL,
    trainable = F,
    # one of "rn50", "rn18", "danQ":
    enc = "rn50") {
    if (enc %in% c("rn50", "rn18")) {
      outlayer <- ifelse(enc == "rn50", 181, 77)
    } else if (enc == "danQ") {
      outlayer <- 14
    }
    
    modl <- keras::load_model_hdf5(trainedmodel, compile = F)
    modr <- keras_model(inputs = modl$input, outputs = modl$get_layer(modl$layers[[outlayer]]$name)$get_output_at(0L))
    
    if (maxlen == 900L) {
      source(paste0("R/architectures/enc_", enc, ".R"), local = T)
      source(paste0("R/architectures/ctx.R"), local = T)
      modl <- encoder(maxlen = maxlen, patchlen =  patchlen)
      model <- keras_model(modl$input, context(modl$output))
      model <- keras_model(modl$input, model$get_layer(model$layers[[outlayer]]$name)$get_output_at(0L))
      set_weights(model, modr$get_weights())
      modr <- model
    }
    
    modf <- modr$output %>% layer_global_average_pooling_1d(name = "global_avg_pool_out")
    modn <- keras_model(modl$input, modf)
    modn$trainable <- trainable
    return(modn)
  }

create_sup_modelLM <- function(maxlen) {
  source("R/architectures/LM_danQ.R", local = T)
  mod <- danQ(maxlen)
  mod <-
    keras_model(
      inputs = mod$input,
      outputs = mod$get_layer(mod$layers[[6]]$name)$get_output_at(0L)
    )
  return(mod)
}

reduce_model_LM <- function(maxlen = 6700L, 
  model = NULL,
  trainable = F) {
  mod <- keras::load_model_hdf5(model, compile = F)
  mod <-
    keras_model(
      inputs = mod$input,
      outputs = mod$get_layer(mod$layers[[6]]$name)$get_output_at(0L)
    )
  mod$trainable <- trainable
  source("R/architectures/LM_danQ.R", local = T)
  mod2 <- danQ(maxlen)
  mod2 <-
    keras_model(
      inputs = mod2$input,
      outputs = mod2$get_layer(mod2$layers[[6]]$name)$get_output_at(0L)
    )
  set_weights(mod2, mod$get_weights())
  return(mod2)
}

getstates <-
  function(reps,
    model,
    generator,
    targetsize = 1,
    outsize = 2048) {
    cat(format(Sys.time(), "%F %R"), ": Generating states\n")
    representations <- list()
    label <- list()
    for (i in seq(reps)) {
      dat <- generator()
      representations[[i]] <-
        as.numeric(model(dat$X %>% tf$convert_to_tensor()))
      label[[i]] <- dat$Y[, targetsize]
      if (i %in% seq(0, reps, by = reps / 10)) {
        cat("-")
      }
    }
    cat("\n",
      format(Sys.time(), "%F %R"),
      " : ",
      reps,
      " states created!\n",
      sep = "")
    if (targetsize == 1) {
      a <- as.factor(t(matrix(
        unlist(label), nrow = 1, ncol = reps
      )))
    } else if (targetsize == 919) {
      a <- t(matrix(unlist(label), nrow = 919, ncol = reps))
    }
    list(data.matrix(t(
      matrix(unlist(representations), nrow = outsize, ncol = reps)
    )), a)
  }


getstatesmat <-
  function(model,
    mat,
    targetsize = 1,
    outsize = 2048,
    maxlen = 6700L) {
    cat(format(Sys.time(), "%F %R"), ": Generating states\n")
    representations <- list()
    label <- list()
    j <- 1
    tf <- import('tensorflow')
    for (i in seq_along(mat[[1]])) {
      if (targetsize == 919) {
        representations[[i]] <-
          as.numeric(model(mat[[1]][[i]][,1:900,] %>% tf$convert_to_tensor() %>% tf$reshape(list(1L, maxlen, 4L))))
      } else{
        representations[[i]] <-
          as.numeric(model(mat[[1]][[i]] %>% tf$convert_to_tensor() %>% tf$reshape(list(1L, maxlen, 4L))))
      }
      label[[i]] <- mat[[2]][[i]]
      if (i %in% ceiling(seq(0, length(mat[[1]]), by = length(mat[[1]]) / 10))) {
        cat(format(Sys.time(), "%F %R"), j*10, "% done \n")
        j <- j + 1
      }
    }
    cat("\n",
      format(Sys.time(), "%F %R"),
      " : ",
      length(mat[[1]]),
      " states created!\n",
      sep = "")
    if (targetsize == 1) {
      a <- as.factor(t(matrix(
        unlist(label), nrow = targetsize, ncol = length(mat[[1]])
      )))
    } else {
      a <-
        t(matrix(unlist(label), nrow = targetsize, ncol = length(mat[[1]])))
    }
    list(data.matrix(t(
      matrix(
        unlist(representations),
        nrow = outsize,
        ncol = length(mat[[1]])
      )
    )), a)
  }


runglmnet <-
  function(datpath,
    targetpath,
    targetname,
    alpha,
    maxlen,
    threeblocks,
    batch.size,
    reps) {
    time  <- format(Sys.time(), "%y%m%d_%H%M%S")
    model <-
      reduce_model(trained_model, threeblocks, trainable = F, maxlen, batch.size)
    
    if (targetname == "GRAM") {
      fas <- fastaLabelGenerator(
        corpus.dir = datpath,
        batch.size = 1,
        maxlen = maxlen,
        randomFiles = T,
        step = maxlen,
        max_samples = 10,
        target_from_csv = targetpath,
        sample_by_file_size = T
      )
    } else if (targetname == "SPECIES") {
      initializeGenerators(
        directories = datpath,
        batch.size = 1,
        maxlen = maxlen,
        randomFiles = T,
        step = maxlen,
        max_samples = 10,
        sample_by_file_size = T
      )
      fastrain <-
        labelByFolderGeneratorWrapper(F, 1, NULL, 1, datpath, NULL, maxlen)
    }
    
    a <- getstates(reps, model, fas)
    
    if (alpha == 1) {
      cat(
        format(Sys.time(), "_%y%m%d_%H%M%S"),
        ": Starting Lasso Regression with",
        reps,
        "samples."
      )
      save <- paste0("model_results/", targetname, "_L1")
    }
    if (alpha == 0) {
      cat(
        format(Sys.time(), "_%y%m%d_%H%M%S"),
        ": Starting Ridge Regression with",
        reps,
        "samples."
      )
      save <- paste0("model_results/", targetname, "_L2")
    }
    
    cv.glmnet <-
      glmnet::cv.glmnet(
        x = a[[1]],
        y = a[[2]],
        type.measure = "class",
        # dviance/AUC
        nfolds = 5,
        #10? / repreated cv
        alpha = alpha,
        # alpha = 1: lasso L1, alpha = 2: ridge L2
        family = "binomial",
        trace.it = 1
      )
    
    c <- coef(cv.glmnet, s = 'lambda.min', exact = TRUE)
    inds <- which(as.numeric(c) != 0)
    c.mat <- as.matrix(as.numeric(c))
    message(paste0(length(which(c.mat[-1, ] > 0)), targetname, "-specific neurons found!"))
    variables <- row.names(c)[inds]
    variables <- variables[variables %ni% '(Intercept)']
    write.table(
      c.mat,
      file = paste0(save, "_coef_", time, ".tsv"),
      sep = "\t",
      quote = F
    )
    saveRDS(cv.glmnet,
      paste0(save, "_glmnet_", time, ".rds"))
    return(cv.glmnet)
  }
