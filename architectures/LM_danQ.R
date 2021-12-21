danQ <- function(maxlen) {
  model <-  keras_model_sequential()
  model %>%
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
    layer_dense(4, activation = "softmax")
}
