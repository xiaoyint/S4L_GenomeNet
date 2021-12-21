encoder <- function(maxlen = NULL,
  patchlen = NULL,
  nopatches = NULL,
  eval = FALSE) {
  if (is.null(nopatches)) {
    source("R/help/calc_help.R")
    nopatches <- nopatchescalc(patchlen, maxlen, patchlen * 0.4)
  }
  inp <- layer_input(shape = c(maxlen, 4))
  stridelen <- as.integer(0.4 * patchlen)
  createpatches <- inp %>%
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
  
  danQ <- createpatches %>%
    layer_conv_1d(
      input_shape = c(maxlen, 4L),
      filters = 320L,
      kernel_size = 26L,
      activation = "relu"
    ) %>%
    layer_max_pooling_1d(pool_size = 13L, strides = 13L) %>%
    layer_dropout(0.2) %>%
    layer_lstm(units = 320, return_sequences = T) %>%
    layer_dropout(0.5) %>%
    layer_flatten() %>%
    layer_dense(925, activation = "relu")
  patchesback <- danQ %>%
    tf$reshape(list(-1L, tf$cast(nopatches, tf$int16), 925L))
  keras_model(inp, patchesback)
}
