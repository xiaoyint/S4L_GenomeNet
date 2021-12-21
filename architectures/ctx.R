context <- function(latents) {
  cres <- latents
  cres_dim = cres$shape
  predictions <-
    cres %>%
    layer_lstm(
      return_sequences = T,
      units = 256,  # WAS: 2048,
      name = paste("context_LSTM_1",
                   sep = ""),
      activation = "relu"
    )
  return(predictions)
}
