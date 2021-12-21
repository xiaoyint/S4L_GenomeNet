library(checkmate) # object type check

########################################################################################################
###################################### Checkpoints saving function #####################################
########################################################################################################

savechecks <- function(cp, runname, model, optimizer, history) {
  ## define path for saved objects
  modpath <-
    paste("model_results/models", runname, cp , sep = "/")
  ## save model object
  model %>% save_model_hdf5(paste0(modpath, "mod_temp.h5"))
  file.rename(paste0(modpath, "mod_temp.h5"),
    paste0(modpath, "mod.h5"))
  ## save optimizer object
  np$save(
    paste0(modpath, "opt.npy"),
    np$array(backend(FALSE)$batch_get_value(optimizer$weights),
      dtype = "object"),
    allow_pickle = TRUE
  )
  ## save history object
  saveRDS(history, paste0(modpath, "history_temp.rds"))
  file.rename(paste0(modpath, "history_temp.rds"),
    paste0(modpath, "history.rds"))
  ## print when finished
  cat(paste0("---------- New ", cp, " model saved\n"))
}

########################################################################################################
############################################ Step function #############################################
########################################################################################################

modelstep <-
  function(trainvaldat,
    model,
    approach,
    ACGT,
    training = F,
    task = NULL,
    auc = NULL) {
    if (approach == "self-supervised-cpc") {
      ## get batch
      a <- trainvaldat$X %>% tf$convert_to_tensor()
      ## get complement if AGCT model
      if (ACGT) {
        a_complement <-
          tf$convert_to_tensor(array(as.array(a)[, (dim(a)[2]):1, 4:1], dim = c(dim(a)[1], dim(a)[2], dim(a)[3])))
        a <- tf$concat(list(a, a_complement), axis = 0L)
      }
      ## insert data in model
      model(a, training = training)
    } else if (approach %in% c("supervised", "self-supervised-nextnuc")) {
      if (task == "deepSea") {
        ## get batch
        a <- trainvaldat
        x <- a[[1]][,1:900,] %>% tf$convert_to_tensor()
        y <- a[[2]] %>% tf$convert_to_tensor()
        ## insert data in model
        pred <- model(x, training = training)
        ## calculate loss and accuracy
        loss <- loss_binary_crossentropy(y, pred) %>%
          tf$stack(axis = 0) %>% tf$reduce_mean()
        acc <- metric_binary_accuracy(tf$cast(y, dtype = "double"),
          tf$cast(pred, dtype = "double")) %>%
          tf$stack(axis = 0) %>% tf$reduce_mean()
        auc$update_state(y_true = tf$cast(y, dtype = "double"),
            y_pred = tf$cast(pred, dtype = "double"))
      } else{
        ## get batch
        a <- trainvaldat
        if (task == "BacVir") {
          x <- a$X[1:32,,] %>% tf$convert_to_tensor()
          y <- a$Y[1:32,] %>% tf$convert_to_tensor()
        } else {
          x <- a$X %>% tf$convert_to_tensor()
          y <- a$Y %>% tf$convert_to_tensor()
        }
        ## insert data in model
        pred <- model(x, training = training)
        ## calculate loss and accuracy
        loss <- loss_categorical_crossentropy(y, pred) %>%
          tf$stack(axis = 0) %>% tf$reduce_mean()
        acc <-
          metric_categorical_accuracy(tf$cast(y, dtype = "double"),
            tf$cast(pred, dtype = "double")) %>%
          tf$stack(axis = 0) %>% tf$reduce_mean()
        if (task == "gram") {
        auc$update_state(y_true = tf$cast(y, dtype = "double"),
          y_pred = tf$cast(pred, dtype = "double"))
        }
      }

      if (task != "deepSea") {
        predi <- tf$convert_to_tensor(np$argmax(pred, axis = 1L))
        truth <- tf$convert_to_tensor(np$argmax(y, axis = 1L))
        
        con_mat <-
          tf$math$confusion_matrix(labels = truth, predictions = predi)
        con_mat_norm = np$around(con_mat / tf$math$reduce_sum(con_mat), decimals = 2L)
        
        con_mat_df = pd$DataFrame(con_mat_norm)
        
        figure <- plt$figure(figsize = list(8, 8))
        sns$heatmap(con_mat_df, annot = T, cmap = plt$cm$Blues)
        plt$tight_layout()
        plt$ylabel('True label')
        plt$xlabel('Predicted label')
        
        buf = io$BytesIO()
        plt$savefig(buf, format = 'png')
        
        plt$close(figure)
        buf$seek(0L)
        image = tf$image$decode_png(buf$getvalue(), channels = 4L)
        
        if (training) {
          .GlobalEnv$imaget = tf$expand_dims(image, 0L)
        } else {
          .GlobalEnv$imagev = tf$expand_dims(image, 0L)
        }
      }
      return(c(loss, acc))
    }
  }

modelstepmat <- function(trainvaldat,
  model,
  batch,
  training = F,
  task = NULL) {
  ## get batch
  browser()
  x <- trainvaldat[[1]][[batch]] %>% tf$convert_to_tensor()
  y <- trainvaldat[[2]][[batch]] %>% tf$convert_to_tensor()
  ## insert data in model
  pred <- model(x, training = training)
  if (task == "deepSea") {
    ## calculate loss and accuracy
    loss <- loss_binary_crossentropy(y, pred) %>%
      tf$stack(axis = 0) %>% tf$reduce_mean()
    acc <- metric_binary_accuracy(tf$cast(y, dtype = "double"),
      tf$cast(pred, dtype = "double")) %>%
      tf$stack(axis = 0) %>% tf$reduce_mean()
    # auc <- metric_auc (
    #   tf$cast(y, dtype = "double"),
    #   tf$cast(pred, dtype = "double"),
    #   multi_label = T,
    #   num_labels = 919L
    # ) %>%
    #   tf$stack(axis = 0) %>% tf$reduce_mean()
  } else{
    ## calculate loss and accuracy
    loss <- loss_categorical_crossentropy(y, pred) %>%
      tf$stack(axis = 0) %>% tf$reduce_mean()
    acc <-
      metric_categorical_accuracy(tf$cast(y, dtype = "double"),
        tf$cast(pred, dtype = "double")) %>%
      tf$stack(axis = 0) %>% tf$reduce_mean()
  }
  
  if (task != "deepSea") {
    predi <- tf$convert_to_tensor(np$argmax(pred, axis = 1L))
    truth <- tf$convert_to_tensor(np$argmax(y, axis = 1L))
    
    con_mat <-
      tf$math$confusion_matrix(labels = truth, predictions = predi)
    con_mat_norm = np$around(con_mat / tf$math$reduce_sum(con_mat), decimals = 2L)
    
    con_mat_df = pd$DataFrame(con_mat_norm)
    
    figure <- plt$figure(figsize = list(8, 8))
    sns$heatmap(con_mat_df, annot = T, cmap = plt$cm$Blues)
    plt$tight_layout()
    plt$ylabel('True label')
    plt$xlabel('Predicted label')
    
    buf = io$BytesIO()
    plt$savefig(buf, format = 'png')
    
    plt$close(figure)
    buf$seek(0L)
    image = tf$image$decode_png(buf$getvalue(), channels = 4L)
    
    if (training) {
      .GlobalEnv$imaget = tf$expand_dims(image, 0L)
    } else {
      .GlobalEnv$imagev = tf$expand_dims(image, 0L)
    }
  }
  return(c(loss, acc))
}

########################################################################################################
################################## Reading Pretrained Model function ###################################
########################################################################################################

ReadOpt <- function(pretrained_model) {
  ## Read configuration
  optconf <-
    readRDS(paste(sub("/[^/]+$", "", pretrained_model),
      "optconfig.rds",
      sep = "/"))
  ## Read optimizer
  optimizer <- tf$optimizers$Adam$from_config(optconf)
  # Initialize optimizer
  with(
    backend()$name_scope(optimizer$`_name`),
    with(tf$python$framework$ops$init_scope(), {
      optimizer$iterations
      optimizer$`_create_hypers`()
      optimizer$`_create_slots`(model$trainable_weights)
    })
  )
  # Read optimizer weights
  wts2 <-
    np$load(paste(
      sub("/[^/]+$", "", pretrained_model),
      "/",
      tail(str_remove(
        strsplit(pretrained_model, "/")[[1]], "mod.h5"
      ), 1),
      "opt.npy",
      sep = ""
    ), allow_pickle = TRUE)
  
  # Set optimizer weights
  optimizer$set_weights(wts2)
  return(optimizer)
}

########################################################################################################
########################################### Build CPC model ############################################
########################################################################################################

buildCPC <- function(resnet_lesschannels,
  resnet_threeblocks,
  GeneSeqConfig,
  CPCconfig) {
  ## Build encoder
  # if specified, create smaller resnet
  if (resnet_lesschannels | resnet_threeblocks) {
    enc <- encoder(GeneSeqConfig,
      lesschannels = resnet_lesschannels,
      threeblocks = resnet_threeblocks)
    # if not, build encoder normally
  } else {
    enc <-
      encoder(GeneSeqConfig)
  }
  
  ## Build model
  model <- keras_model(enc$input, cpcloss(enc$output, CPCconfig))
}

LRstop <- function(lr_schedule) {
  # cosine annealing
  if ("cosine_annealing" %in% lr_schedule) {
    if (!isTRUE(all.equal(sort(names(lr_schedule)), sort(
      c("schedule", "lrmin", "lrmax", "restart", "mult")
    )))) {
      stop(
        "Please define lrmin, lrmax, restart, and mult within the list to use cosine annealing"
      )
    }
    # step decay
  } else if ("step_decay" %in% lr_schedule) {
    if (!isTRUE(all.equal(sort(names(lr_schedule)), sort(
      c("schedule", "lrmax", "newstep", "mult")
    )))) {
      stop("Please define lrmax, newstep, and mult within the list to use step decay")
    }
    # exponential decay
  } else if ("exp_decay" %in% lr_schedule) {
    if (!isTRUE(all.equal(sort(names(lr_schedule)), sort(c(
      "schedule", "lrmax", "mult"
    ))))) {
      stop("Please define lrmax, and mult within the list to use exponential decay")
    }
  }
}

getEpochLR <- function(lr_schedule, epoch, lr_schedule_shift) {
  if (lr_schedule$schedule == "cosine_annealing") {
    # cosine annealing
    sgdr(
      lrmin = lr_schedule$lrmin,
      restart = lr_schedule$restart,
      lrmax = lr_schedule$lrmax,
      mult = lr_schedule$mult,
      epoch = epoch + lr_schedule_shift
    )
  } else if (lr_schedule$schedule == "step_decay") {
    # step decay
    stepdecay(
      newstep = lr_schedule$newstep,
      lrmax = lr_schedule$lrmax,
      mult = lr_schedule$mult,
      epoch = epoch + lr_schedule_shift
    )
    
  } else if (lr_schedule$schedule == "exp_decay") {
    # exponential decay
    exp_decay(
      lrmax = lr_schedule$lrmax,
      mult = lr_schedule$mult,
      epoch = epoch + lr_schedule_shift
    )
  }
}

TB_loss_acc <- function(writer, loss, acc, epoch) {
  with(writer$as_default(), {
    tf$summary$scalar('epoch_loss',
      loss$result(),
      step = tf$cast(epoch, "int64"))
    tf$summary$scalar('epoch_accuracy',
      acc$result(),
      step = tf$cast(epoch, "int64"))
  })
}

########################################################################################################
########################################### Parameter Lists ############################################
########################################################################################################
#
GenParams <- function(maxlen,
  batch.size,
  step,
  proportion_per_file,
  max_samples) {
  assertInt(maxlen, lower = 1)
  assertInt(batch.size, lower = 1)
  assertInt(step, lower = 1)
  assertInt(max_samples, lower = 1, null.ok = T)
  assertNumber(
    proportion_per_file,
    lower = 0,
    upper = 1,
    null.ok = T
  )
  
  structure(
    list(
      maxlen = maxlen,
      batch.size = batch.size,
      step = step,
      proportion_per_file = proportion_per_file,
      max_samples = max_samples
    ),
    class = "Params"
  )
}
#
GenTParams <- function(path,
  randomFilesTrain,
  fileLog,
  seed) {
  assertLogical(randomFilesTrain)
  assertInt(seed)
  
  structure(
    list(
      corpus.dir = path,
      randomFiles = randomFilesTrain,
      fileLog = fileLog,
      seed = seed
    ),
    class = "Params"
  )
}

GenVParams <- function(path.val,
  randomFilesVal) {
  assertLogical(randomFilesVal)
  
  structure(list(corpus.dir = path.val[[1]],
    randomFiles = randomFilesVal),
    class = "Params")
}

# GeneSeqParams <- function(maxlen,
#   patchlen,
#   nopatches) {
#   assertInt(maxlen, lower = 1)
#   assertInt(patchlen, lower = 1)
#   assertInt(nopatches, lower = 1)
#
#   structure(list(
#     maxlen = maxlen,
#     patchlen = patchlen,
#     nopatches = nopatches
#   ),
#     class = "Params")
# }
#
# CPCParams <- function(context,
#   encoder,
#   batch.size,
#   stepsmin,
#   stepsmax,
#   ACGT,
#   k,
#   emb_scale) {
#   assertInt(batch.size, lower = 1)
#   assertInt(stepsmin, lower = 1)
#   assertInt(stepsmax, lower = stepsmin)
#   assertLogical(ACGT)
#   assertInt(k, lower = 1)
#   assertNumber(emb_scale)
#
#   structure(
#     list(
#       context = context,
#       encoder = encoder,
#       batch.size = batch.size,
#       steps_to_ignore = stepsmin,
#       steps_to_predict = stepsmax,
#       ACGT = ACGT,
#       onlysameposition = ACGT,
#       k = k,
#       emb_scale = emb_scale
#     ),
#     class = "Params"
#   )
# }


#####################################################
#
# TrainParams <- function(learning.rate, epoch.size) {
#   assertNumber(learning.rate, lower = 0)
#   assertInt(epoch.size, lower = 1)
#   structure(list(learning.rate = learning.rate, epoch.size = epoch.size),
#     class = "TrainParams")
# }
#
# print.TrainParams <- function(x, ...) {
#   cat(
#     sprintf(
#       "Training config!\nEpoch size: %s\nLearning rate: %s\n",
#       x$epoch.size,
#       x$learning.rate
#     )
#   )
#   invisible(x)
# }
#
#
# cpc.trained <-
#   traincpc1(
#     model,
#     generator.train,
#     generator.validation,
#     cpcparam1 = 1,
#     cpcparam2 = 100
#   )
#
# result <-
#   eval.semisupervised(
#     cpc.trained,
#     generator.train,
#     generator.validation,
#     ssparam1 = 100,
#     ssparam2 = "test"
#   )