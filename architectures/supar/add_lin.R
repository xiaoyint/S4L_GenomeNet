addl <- function(model, outsize = 2, act = "softmax") {
  model1 <- layer_dense(model$output, outsize, activation = act, name = "lin_tar")
  model1 <- keras_model(model$input, model1)
  return(model1)
}
