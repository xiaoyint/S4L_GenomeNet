addl <- function(model, outsize = 2, act = "softmax") {
  model1 <-
    layer_reshape(model$output, list(model$output$shape[2][[1]], 1L)) %>%
    layer_lstm(256) %>%
    layer_dense(outsize, activation = act)
  model1 <- keras_model(model$input, model1)
  return(model1)
}
