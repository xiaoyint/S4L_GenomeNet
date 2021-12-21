addl <- function(model, outsize = 2, act = "softmax") {
  model1 <-
    layer_reshape(model$output, list(model$output$shape[2][[1]], 1L)) %>%
    layer_conv_1d(32, 100, 80) %>%
    layer_conv_1d(16, 5, 4) %>%
    layer_flatten() %>%
    layer_dense(outsize, activation = act)
  model1 <- keras_model(model$input, model1)
  return(model1)
}
