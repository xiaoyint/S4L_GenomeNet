addl <- function(model, outsize = 2, act = "softmax") {
  model1 <- 
    layer_dense(model$output, 1024) %>%
    layer_dense(512) %>%
    layer_dense(outsize, activation = act)
  model1 <- keras_model(model$input, model1)
  return(model1)
}
