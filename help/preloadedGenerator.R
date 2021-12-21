library(deepG)

listGenerator <- function(l, batchmultiplier = 1) {
  i <- 0
  force(l)
  stopifnot(length(l) >= batchmultiplier)
  stopifnot(batchmultiplier %% 1 == 0)
  list(
    gen =
      function() {
        i <<- i + batchmultiplier
        if (i > length(l))
          i <<- batchmultiplier
        if (batchmultiplier == 1) {
          l[[i]]
        } else {
          indices <- names(l[[i]])
          if (is.null(indices)) indices <- seq_along(l[[i]])
          sapply(indices, function(idx) {
            do.call(abind::abind, c(
              list(along = 1),
              lapply(l[seq(to = i, length.out = batchmultiplier)], `[[`, idx)
            ))
          }, simplify = FALSE)
        }
      },
    reset = function()
      i <<- 0
  )
}

preloadedGenerator <- function(generator, loadsteps, ..., batchmultiplier = 1) {
  gen <- generator(...)
  listGenerator(replicate(loadsteps, gen(), simplify = FALSE), batchmultiplier = batchmultiplier)
}

preloadPLG <- function(generator, loadsteps, file, ...) {
  gen <- generator(...)
  saveRDS(replicate(loadsteps, gen(), simplify = FALSE), file)
  invisible(NULL)
}

readPLG <- function(file, batchmultiplier = 1) {
  listGenerator(readRDS(file), batchmultiplier = batchmultiplier)
}

readPLGpar <-
  function(preloadGeneratorpath,
    batch.size,
    maxlen,
    max_samples = 8,
    seed,
    type = NULL, 
    batchmultiplier = 1) {
    readPLG(
      paste0(
        preloadGeneratorpath,
        "/batchsize",
        batch.size,
        "_maxlen",
        maxlen,
        "_max_samples",
        max_samples,
        "_seed_",
        seed,
        type,
        ".rds"
      ),
      batchmultiplier = batchmultiplier
    )
  }

preloadPLG2 <- function(generator, loadsteps, file, ...) {
  l <- list()
  t <- 0
  gen <- generator(...)
  for (i in seq(loadsteps)) {
    l[[i]] <- gen()
    if (i %in% seq(0,loadsteps, by = loadsteps/10)) {
      t <- t + 10
      print(paste0(Sys.time(), ": ", t, "% done!"))
    }
  }
  saveRDS(l, file)
}



# fg <- fastaFileGenerator(
#   path.val,
#   batch.size = 3,
#   maxlen = 5,
#   step = 5,
#   max_samples = 7, ######16
#   randomFiles = TRUE,
#   seed = 888
# )
#
# fg()
#
# pfg <- preloadedGenerator(fastaFileGenerator, 10,
#   path.val,
#   batch.size = 3,
#   maxlen = 5,
#   step = 5,
#   max_samples = 7,
#   randomFiles = TRUE,
#   seed = 888
#
# )
# pfg <- readPLG("/tmp/x.rds")
# pfg <- readPLG("/tmp/batchsize32_maxlen8100_max_samples16_seed_888.rds")
# pfg$gen()
#
# pfg$reset()
