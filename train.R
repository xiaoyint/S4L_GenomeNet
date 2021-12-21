library(keras)
library(deepG)
library(magrittr)
library(tensorflow)
library(tfdatasets)
library(tfautograph)
library(purrr)
library(readr)
library(stringr)

Train <-
  function(#### Generator settings ####
    path,
    path.val = NULL,
    maxlen = NULL,
    patchlen = NULL,
    nopatches = NULL,
    step = NULL,
    batch.size = 32,
    batchmultiplier = 1,
    proportion_per_file = NULL,
    randomFilesTrain = F,
    randomFilesVal = F,
    max_samples = NULL,
    preloadGeneratorpath = NULL,
    file_filter = NULL,
    numberOfFiles = NULL,
    # number of files per group in train
    seed = 1234,
    #stride = 0.4,
    # proportional for patchlen
    #### Model settings ####
    approach = "self-supervised-cpc",
    # or "self-supervised-nextnuc" or "supervised"
    sup_nn_model = NULL,
    pretrained_model = NULL,
    task = "CPC",
    # deepSea, folder
    nogenmat = F,
    ### cpc functions ###
    encoder = NULL,
    context = NULL,
    cpcloss = NULL,
    ### Supervised model settings ###
    targets = NULL,
    #### Training Settings ####
    start_epoch = 1,
    epochs = 100,
    steps.per.epoch = 2000,
    learningrate = 0.0004,
    lr_schedule = NULL,
    lr_schedule_shift = 0,
    validation_split = 0.2,
    k = 5,
    #### Hyperparameters CPC ####
    ACGT = F,
    resnet_lesschannels = F,
    resnet_threeblocks = F,
    stepsmin = 2,
    stepsmax = 3,
    emb_scale = 0.1,
    #### Callbacks ####
    run.name,
    tensorboard.log = NULL,
    savemodels = T,
    saveTB = T,
    fileLog = T,
    fileeval = F,
    testrun = F,
    valinspect = F) {
    # Stride is default 0.4 x patchlen FOR NOW
    stride <- 0.4
    
    source("R/help/train_help.R", local = T)
    np = import("numpy", convert = F)
    sns <- import('seaborn')
    pd <- import('pandas')
    plt <- import('matplotlib.pyplot')
    sklearn <- import('sklearn')
    io <- import('io')
    
    ########################################################################################################
    ############################### Warning messages if wrong initialization ###############################
    ########################################################################################################
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model specification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    ## Three options:
    ## 1. Define Maxlen and Patchlen
    ## 2. Define Number of patches and Patchlen
    ## ---> in both cases the respectively missing value will be calculated
    ## 3. Pretrained model is giving specs
    ## error if none of those is fulfilled
    
    if (is.null(pretrained_model) &&
        approach == "self-supervised-cpc") {
      ## If no pretrained model, patchlen has to be defined
      if (is.null(patchlen)) {
        stop("Please define patchlen")
      }
      ## Either maxlen or number of patches is needed
      if (is.null(maxlen) & is.null(nopatches)) {
        stop("Please define either maxlen or nopatches")
        ## the respectively missing value will be calculated
      } else if (is.null(maxlen) & !is.null(nopatches)) {
        maxlen <- (nopatches - 1) * (stride * patchlen) + patchlen
      } else if (!is.null(maxlen) & is.null(nopatches)) {
        nopatches <-
          as.integer((maxlen - patchlen) / (stride * patchlen) + 1)
      }
      ## if step is not defined, we do not use overlapping sequences
      if (is.null(step)) {
        step = maxlen
      }
    } else if (!is.null(pretrained_model)) {
      specs <-
        readRDS(paste(
          sub("/[^/]+$", "", pretrained_model),
          "modelspecs.rds",
          sep = "/"
        ))
      patchlen          <- specs$patchlen
      maxlen            <- specs$maxlen
      nopatches         <- specs$nopatches
      stride            <- specs$stride
      step              <- specs$step
      #batch.size        <- specs$batch.size
      #steps.per.epoch   <- specs$steps.per.epoch
      #validation_split  <- specs$validation_split
      k                 <- specs$k
      emb_scale         <- specs$emb_scale
    }
    if (approach == "supervised") {
      if (!is.null(sup_nn_model)) {
        maxlen <- patchlen
      }
    }
    
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Learning rate schedule ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    ## If learning rate schedule is wanted, all necessary parameters must be given
    # cosine annealing
    LRstop(lr_schedule)
    ########################################################################################################
    #################################### Preparation: Data, paths metrics ##################################
    ########################################################################################################
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File count ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    if (!nogenmat) {
      if (is.null(file_filter)) {
        if (is.null(numberOfFiles)) {
          if (is.list(path)) {
            num_files <- 0
            for (i in seq_along(path)) {
              num_files <- num_files + length(list.files(path[[i]]))
            }
          } else {
            num_files <- length(list.files(path))
          }
        } else {
          num_files <- numberOfFiles * length(path)
        }
      } else {
        num_files <- length(file_filter[1])
      }
    }
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Path definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    ## If testrun, place in specific folder
    if (testrun) {
      if (fileeval) {
        runname <-
          paste("testruns/fileeval/",
            run.name,
            format(Sys.time(), "_%y%m%d_%H%M%S"),
            sep = "")
      } else {
        runname <-
          paste0("testruns/",
            run.name,
            format(Sys.time(), "_%y%m%d_%H%M%S"))
      }
    } else {
      runname <-
        paste0(run.name, format(Sys.time(), "_%y%m%d_%H%M%S"))
    }
    
    ## Create folder for model
    if (savemodels) {
      dir.create(paste("model_results/models", runname, sep = "/"))
    }
    
    ## Create folder for filelog
    if (fileLog) {
      fileLog <-
        paste("model_results/log", runname, "filelog.csv", sep = "/")
      dir.create(paste("model_results/log", runname, sep = "/"))
    } else {
      fileLog <- NULL
    }
    
    if (task %in% c("CPC", "gram", "MOT")) {
      GenConfig <-
        GenParams(maxlen, batch.size, step, proportion_per_file, max_samples)
      GenTConfig <-
        GenTParams(path, randomFilesTrain, fileLog, seed)
      if (is.null(preloadGeneratorpath)) {
        GenVConfig <- GenVParams(path.val, randomFilesVal)
      }
    }
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Creation of generators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    if (!nogenmat) {
      cat(format(Sys.time(), "%F %R"), ": Preparing the data\n")
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training Generator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      if (approach %in% c("self-supervised-cpc", "self-supervised-nextnuc")) {
        fastrain <-
          do.call(fastaFileGenerator,
            c(GenConfig, GenTConfig, file_filter = file_filter[1]))
      } else if (approach == "supervised") {
        if (task == "deepSea") {
          fastrain <- gen_rds(path, batch.size, fileLog)
        } else if (task %in% c("BacVir", "gram", "MOT")) {
          initializeGenerators(
            path,
            batch.size = batch.size,
            maxlen = maxlen,
            randomFiles = randomFilesTrain,
            step = step,
            seed = seed,
            numberOfFiles = numberOfFiles,
            fileLog = fileLog,
            max_samples = max_samples
          )
          fastrain <-
            labelByFolderGeneratorWrapper(F, batch.size, NULL, batch.size, path, NULL, maxlen)
        }
      }
      
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Validation Generator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      if (!is.null(preloadGeneratorpath)) {
        source("R/help/preloadedGenerator.R")
        fasval <-
          readPLGpar(preloadGeneratorpath, batch.size/batchmultiplier, maxlen, 8, seed,
            batchmultiplier = batchmultiplier)
      } else{
        if (approach %in% c("self-supervised-cpc", "self-supervised-nextnuc")) {
          fasval <-
            do.call(
              fastaFileGenerator,
              c(
                GenConfig,
                GenVConfig,
                seed = seed,
                file_filter = file_filter[2]
              )
            )
        } else if (approach == "supervised") {
          #browser()
          if (task == "deepSea") {
            fasval <- gen_rds(path.val, batch.size)
          } else if (task %in% c("BacVir", "gram", "MOT")) {
            bs <- ifelse(task == "BacVir", 33, 32)
            initializeGenerators(
              path.val,
              batch.size = bs,
              maxlen = maxlen,
              randomFiles = randomFilesVal,
              step = step,
              seed = seed,
              max_samples = max_samples,
              val = T
            )
            fasval <-
              labelByFolderGeneratorWrapper(T,
                batch.size,
                NULL,
                bs,
                path.val,
                NULL,
                maxlen)
          }
        }
      }
    }
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Creation of metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    cat(format(Sys.time(), "%F %R"), ": Preparing the metrics\n")
    train_loss <- tf$keras$metrics$Mean(name = 'train_loss')
    val_loss <- tf$keras$metrics$Mean(name = 'val_loss')
    train_acc <- tf$keras$metrics$Mean(name = 'train_acc')
    val_acc <- tf$keras$metrics$Mean(name = 'val_acc')
    if (approach == "supervised") {
      if (task == "deepSea") {
        train_auc <-
          metric_auc(
            multi_label = T,
            num_labels = 919L,
            from_logits = T
          )
        val_auc <-
          metric_auc(
            multi_label = T,
            num_labels = 919L,
            from_logits = T
          )
      } else if (task %in% c("gram", "MOT")) {
        train_auc <-
          metric_auc()
        val_auc <-
          metric_auc()
      }
    } else {
      train_auc <- NULL
      val_auc <- NULL
    }
    ####~~~~~~~~~~~~~~~~~~~~~~~~ Additional generators for validation inspection ~~~~~~~~~~~~~~~~~~~~~~~####
    if (!nogenmat) {
      # create if no preloaded specified
      if (valinspect) {
        cat(format(Sys.time(), "%F %R"), ": Preparing the valinspect data\n")
        seeds <- c(888, 999, 111, 222)
        fasvali <- list()
        val_lossi <- list()
        val_acci <- list()
        
        if (is.null(preloadGeneratorpath)) {
          for (j in seeds) {
            fasvali[[match(j, seeds)]] <-
              do.call(
                fastaFileGenerator,
                c(
                  GenConfig,
                  GenVConfig,
                  seed = j,
                  file_filter = file_filter[2]
                )
              )
          }
          
          # read if preloaded specified
        } else {
          source("R/help/preloadedGenerator.R")
          for (j in seeds) {
            fasvali[[match(j, seeds)]] <-
              readPLGpar(preloadGeneratorpath,
                batch.size/batchmultiplier,
                maxlen,
                max_samples,
                seed,
                batchmultiplier = batchmultiplier)
          }
        }
        ####~~~~~~~~~~~~~~~~~~~~~~~~ Additional metrics for validation inspection ~~~~~~~~~~~~~~~~~~~~~~~~####
        for (j in 1:4) {
          val_lossi[[j]] <- tf$keras$metrics$Mean(name = 'val_loss')
          val_acci[[j]] <- tf$keras$metrics$Mean(name = 'val_acc')
        }
      }
    }
    ########################################################################################################
    ###################################### History object preparation ######################################
    ########################################################################################################
    
    .GlobalEnv$history <- list(
      params = list(
        batch_size = batch.size,
        epochs = 0,
        steps = steps.per.epoch,
        samples = steps.per.epoch * batch.size,
        verbose = 1,
        do_validation = T,
        metrics = c("loss", "accuracy", "val_loss", "val_accuracy")
      ),
      metrics = list(
        loss = c(),
        accuracy = c(),
        val_loss = c(),
        val_accuracy = c()
      )
    )
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~ Additional metrics for validation inspection ~~~~~~~~~~~~~~~~~~~~~~~~~####
    if (valinspect) {
      .GlobalEnv$history$params$metrics <-
        append(
          .GlobalEnv$history$params$metrics,
          c(
            "val_loss2",
            "val_accuracy2",
            "val_loss3",
            "val_accuracy3",
            "val_loss4",
            "val_accuracy4",
            "val_loss5",
            "val_accuracy5"
            # do.call(paste0, expand.grid(c("val_loss","val_acc"), c(2,3,4,5)))
          )
        )
      .GlobalEnv$history$metrics <-
        append(
          .GlobalEnv$history$metrics,
          list(
            val_loss2 = c(),
            val_accuracy2 = c(),
            val_loss3 = c(),
            val_accuracy3 = c(),
            val_loss4 = c(),
            val_accuracy4 = c(),
            val_loss5 = c(),
            val_accuracy5 = c()
          )
        )
    }
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reformat to S3 object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    class(.GlobalEnv$history) <- "keras_training_history"
    
    ########################################################################################################
    ############################################ Model creation ############################################
    ########################################################################################################
    if (is.null(pretrained_model) ) {
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Supervised: allocate model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      if (approach %in% c("supervised", "self-supervised-nextnuc")) {
      # define model
      model <- sup_nn_model
      
      ## Build optimizer
      optimizer <- optimizer_adam(
        learning_rate = learningrate,
        beta_1 = 0.8,
        epsilon = 10 ^ -8,
        decay = 0.999,
        clipnorm = 0.01
      )} else {
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsupervised Build from scratch ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      cat(format(Sys.time(), "%F %R"), ": Creating the model\n")
      ## Build encoder
      # if specified, create smaller resnet
      if (resnet_lesschannels | resnet_threeblocks) {
        enc <- encoder(
          maxlen = maxlen,
          patchlen = patchlen,
          nopatches = nopatches,
          lesschannels = resnet_lesschannels,
          threeblocks = resnet_threeblocks
        )
        # if not, build encoder normally
      } else {
        enc <-
          encoder(maxlen = maxlen,
            patchlen = patchlen,
            nopatches = nopatches)
      }
      
      ## Build model
      model <-
        keras_model(
          enc$input,
          cpcloss(
            enc$output,
            context,
            batch.size = batch.size,
            steps_to_ignore = stepsmin,
            steps_to_predict = stepsmax,
            ACGT = ACGT,
            onlysameposition = ACGT,
            k = k,
            emb_scale = emb_scale
          )
        )
      
      ## Build optimizer
      optimizer <- optimizer_adam(
        learning_rate = learningrate,
        beta_1 = 0.8,
        epsilon = 10 ^ -8,
        decay = 0.999,
        clipnorm = 0.01
      )
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsupervised Read if pretrained model given ~~~~~~~~~~~~~~~~~~~~~~~~~####
      } 
    } else {
      cat(format(Sys.time(), "%F %R"), ": Loading the trained model.\n")
      ## Read model
      model <- load_model_hdf5(pretrained_model, compile = F)
      optimizer <- ReadOpt(pretrained_model)
      optimizer$learning_rate$assign(learningrate)
    } 
    
    # else {
    #   stop("Please define model to train")
    # }
    
    ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving necessary model objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
    if (savemodels) {
      ## optimizer configuration
      saveRDS(
        optimizer$get_config(),
        paste("model_results/models", runname, "optconfig.rds", sep = "/")
      )
      ## model parameters
      saveRDS(
        list(
          maxlen = maxlen,
          patchlen = patchlen,
          stride = stride,
          nopatches = nopatches,
          step = step,
          batch.size = batch.size,
          epochs = epochs,
          steps.per.epoch = steps.per.epoch,
          validation_split = validation_split,
          max_samples = max_samples,
          k = k,
          emb_scale = emb_scale,
          learningrate = learningrate,
          testrun = as.character(testrun)
        ),
        paste("model_results/models", runname, "modelspecs.rds", sep = "/")
      )
    }
    
    ########################################################################################################
    ######################################## Tensorboard connection ########################################
    ########################################################################################################
    
    if (saveTB) {
      if (is.null(tensorboard.log)) {
        stop("Please define tensorboard log folder or set saveTB to false")
      }
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialize Tensorboard writers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      logdir <- tensorboard.log
      writertrain <-
        tf$summary$create_file_writer(file.path(logdir, runname, "/train"))
      writerval <-
        tf$summary$create_file_writer(file.path(logdir, runname, "/validation"))
      
      if (valinspect) {
        writervali <- list()
        for (i in 1:4) {
          writervali[[i]] = tf$summary$create_file_writer(file.path(logdir, runname, paste0("/validation", i + 1)))
        }
      }
      ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Write parameters to Tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
      tftext <-
        lapply(as.list(match.call())[-1][-c(1, 2)], function(x)
          ifelse(all(nchar(deparse(
            eval(x)
          )) < 20) && !is.null(eval(x)), eval(x), deparse(x)))
      
      with(writertrain$as_default(), {
        tf$summary$text("Specification",
          paste(
            names(tftext),
            tftext,
            sep = " = ",
            collapse = "  \n"
          ),
          step = 0L)
      })
    }
    
    ########################################################################################################
    ######################################## Training loop function ########################################
    ########################################################################################################
    
    train_val_loop <-
      function(batches = steps.per.epoch, epoch, validation_split) {
        ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start of loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
        for (i in c("train", "val")) {
          if (i == "val") {
            ## Calculate steps for validation
            batches <- ceiling(batches * validation_split)
          }
          
          for (b in seq(batches)) {
            if (i == "train") {
              ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
              ## If Learning rate schedule specified, calculate learning rate for current epoch
              if (!is.null(lr_schedule)) {
                source("R/help/calc_help.R")
                optimizer$learning_rate$assign(getEpochLR(lr_schedule, epoch, lr_schedule_shift))
              }
              ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimization step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
              
              with(tf$GradientTape() %as% tape, {
                if (!nogenmat) {
                  out <-
                    modelstep(fastrain(),
                      model,
                      approach,
                      ACGT,
                      T,
                      task,
                      train_auc)
                } else {
                  out <- modelstepmat(path, model, b, T, task)
                }
                l <- out[1]
                acc <- out[2]
              })
              
              gradients <-
                tape$gradient(l, model$trainable_variables)
              optimizer$apply_gradients(purrr::transpose(list(
                gradients, model$trainable_variables
              )))
              train_loss(l)
              train_acc(acc)
              
            } else {
              ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Validation step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
              if (is.null(preloadGeneratorpath)) {
                if (!nogenmat) {
                  out <-
                    modelstep(fasval(),
                      model,
                      approach,
                      ACGT,
                      F,
                      task,
                      val_auc)
                } else {
                  out <- modelstepmat(path.val, model, b, F, task)
                }
              } else {
                out <-
                  modelstep(fasval$gen(),
                    model,
                    approach,
                    ACGT,
                    F,
                    task,
                    val_auc)
              }
              l <- out[1]
              acc <- out[2]
              val_loss(l)
              val_acc(acc)
              
              ## additional validation steps if validation evaluation enabled
              if (valinspect) {
                outi <- list()
                for (i in 1:4) {
                  if (is.null(preloadGeneratorpath)) {
                    outi[[i]] <-
                      modelstep(fasvali[[i]](),
                        model,
                        approach,
                        ACGT,
                        F,
                        task,
                        val_auc)
                  } else {
                    outi[[i]] <-
                      modelstep(fasvali[[i]]$gen(),
                        model,
                        approach,
                        ACGT,
                        F,
                        task)
                  }
                  val_lossi[[i]](outi[[i]][1])
                  val_acci[[i]](outi[[i]][2])
                }
              }
            }
            
            ## Print status of epoch
            if (b %in% seq(0, batches, by = batches / 10)) {
              cat("-")
            }
          }
          
          ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of Epoch ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~####
          if (i == "train") {
            ## Training step
            # Write epoch result metrics value to tensorboard
            if (saveTB) {
              TB_loss_acc(writertrain, train_loss, train_acc, epoch)
              with(writertrain$as_default(), {
                tf$summary$scalar('epoch_lr',
                  optimizer$learning_rate,
                  step = tf$cast(epoch, "int64"))
                if (!nogenmat) {
                  tf$summary$scalar(
                    'training files seen',
                    nrow(
                      read_csv(
                        fileLog,
                        col_names = F,
                        col_types = cols()
                      )
                    ) / num_files,
                    step = tf$cast(epoch, "int64")
                  )
                }
              })
              if (approach %in% c("supervised", "self-supervised-nextnuc") &&
                  task != "deepSea") {
                with(writertrain$as_default(), {
                  tf$summary$image("confusion matrix train",
                    imaget,
                    step = tf$cast(epoch, "int64"))
                })
              }
              if (task %in% c("deepSea", "gram", "MOT")) {
                with(writertrain$as_default(), {
                  tf$summary$scalar('epoch_auc',
                    train_auc$result(),
                    step = tf$cast(epoch, "int64"))
                })
              }
            }
            # Print epoch result metric values to console
            tf$print(" Train Loss",
              train_loss$result(),
              ", Train Acc",
              train_acc$result())
            
            # Save epoch result metric values to history object
            .GlobalEnv$history$params$epochs <- epoch
            .GlobalEnv$history$metrics$loss[epoch] <-
              as.double(train_loss$result())
            .GlobalEnv$history$metrics$accuracy[epoch]  <-
              as.double(train_acc$result())
            
            # Reset states
            train_loss$reset_states()
            train_acc$reset_states()
            if (task %in% c("deepSea", "gram", "MOT")) {
              train_auc$reset_states()
            }
            
          } else {
            ## Validation step
            # Write epoch result metrics value to tensorboard
            if (saveTB) {
              TB_loss_acc(writerval, val_loss, val_acc, epoch)
              if (approach %in% c("supervised", "self-supervised-nextnuc") &&
                  task != "deepSea") {
                with(writerval$as_default(), {
                  tf$summary$image("confusion matrix validation",
                    imagev,
                    step = tf$cast(epoch, "int64"))
                })
              }
              if (task %in% c("deepSea", "gram", "MOT")) {
                with(writerval$as_default(), {
                  tf$summary$scalar('epoch_auc',
                    val_auc$result(),
                    step = tf$cast(epoch, "int64"))
                })
              }
              if (valinspect) {
                for (i in 1:4) {
                  TB_loss_acc(writervali[[i]], val_lossi[[i]], val_acci[[i]], epoch)
                }
              }
            }
            
            # Print epoch result metric values to console
            tf$print(" Validation Loss",
              val_loss$result(),
              ", Validation Acc",
              val_acc$result())
            
            # save results globally for best model saving condition
            if (b == max(seq(batches))) {
              .GlobalEnv$eploss[[epoch]] <- as.double(val_loss$result())
              .GlobalEnv$epacc[[epoch]] <-
                as.double(val_acc$result())
            }
            
            # Save epoch result metric values to history object
            .GlobalEnv$history$metrics$val_loss[epoch] <-
              as.double(val_loss$result())
            .GlobalEnv$history$metrics$val_accuracy[epoch]  <-
              as.double(val_acc$result())
            
            # Reset states
            val_loss$reset_states()
            val_acc$reset_states()
            if (task %in% c("deepSea", "gram", "MOT")) {
              val_auc$reset_states()
            }
            if (!is.null(preloadGeneratorpath)) {
              fasval$reset()
            }
            
            ## additional validation step results if validation evaluation enabled
            if (valinspect) {
              for (j in 1:4) {
                .GlobalEnv$history$metrics[[(j * 2) + 3]][epoch] <-
                  as.double(val_lossi[[j]]$result())
                .GlobalEnv$history$metrics[[(j * 2) + 4]][epoch] <-
                  as.double(val_acci[[j]]$result())
                val_lossi[[j]]$reset_states()
                val_acci[[j]]$reset_states()
              }
              # reset preloaded Generator to have the same values each epoch
              if (!is.null(preloadGeneratorpath)) {
                for (j in 1:4) {
                  fasvali[[j]]$reset()
                }
              }
            }
          }
        }
      }
    
    ########################################################################################################
    ############################################# Training run #############################################
    ########################################################################################################
    
    # initialize global list of validation results for best model saving condition
    .GlobalEnv$eploss <- list()
    .GlobalEnv$epacc <- list()
    
    if (fileeval) {
      nfiles <- 0
      stime <- Sys.time()
      fileevaldt <- data.frame()
    }
    
    cat(format(Sys.time(), "%F %R"), ": Starting Training\n")
    
    ## Training loop
    for (i in seq(start_epoch, (epochs + start_epoch - 1))) {
      cat(format(Sys.time(), "%F %R"), ": EPOCH", i, " \n")
      
      ## Epoch loop
      train_val_loop(epoch = i, validation_split = validation_split)
      
      ## Save checkpoints
      if (savemodels) {
        # best model (smallest loss)
        if (eploss[[i]] == min(unlist(eploss))) {
          savechecks("best", runname, model, optimizer, history)
        }
        # backup model every 10 epochs
        if (i %% 10 == 0) {
          savechecks("backup", runname, model, optimizer, history)
        }
      }
      if (fileeval) {
        epfile <- list(
          "files_opened" = (nrow(
            read_csv(fileLog,
              col_names = F,
              col_types = cols())
          ) - nfiles),
          "duration" = (difftime(Sys.time(), stime)),
          "max_samples" = max_samples,
          epoch = i
        )
        print(as.data.frame(epfile))
        fileevaldt <-
          rbind(fileevaldt, epfile)
        nfiles <- nrow(read_csv(fileLog,
          col_names = F,
          col_types = cols()))
        stime <- Sys.time()
      }
    }
    
    ########################################################################################################
    ############################################# Final saves ##############################################
    ########################################################################################################
    
    if (savemodels) {
      savechecks(cp = "FINAL", runname, model, optimizer, history)
    }
    if (fileeval) {
      saveRDS(fileevaldt,
        paste("other_dat", runname, "fileeval.rds", sep = "/"))
    }
    if (saveTB) {
      writegraph <-
        tf$keras$callbacks$TensorBoard(file.path(logdir, runname))
      writegraph$set_model(model)
    }
  }