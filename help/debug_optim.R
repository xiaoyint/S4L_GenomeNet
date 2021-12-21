library(keras)
library(deepG)
library(magrittr)
library(tensorflow)
library(tfdatasets)
library(tfautograph)
library(purrr)
library(readr)
source("MAGenomeNet/R/traincpc.R")
source("MAGenomeNet/R/resnet.R")
source("MAGenomeNet/R/context2.R")
source("MAGenomeNet/R/cpcloss.R")

path            = "/genedata/genomes/bacteria_1_3/train/"

fastrain <-
  fastaFileGenerator(
    path,
    batch.size = 8,
    maxlen = 8100,
    step = 8100
  )

optimizer <- optimizer_adam(
  lr = 0.001,
  beta_1 = 0.8,
  epsilon = 10 ^ -8,
  decay = 0.999,
  # .9999
  clipnorm = 0.01
)


enc <- resnet1d(
  maxlen = 8100L,
  patchlen = 500L,
  nopatches = 39L,
  small = TRUE
)

model <-
  keras_model(
    enc$input,
    cpc(
      enc$output,
      context2,
      batch.size = 8
      )
  )

modelstep <- function(trainvaldat) {
  a <- trainvaldat$X %>% tf$convert_to_tensor()
  model(a)
}

with(tf$GradientTape() %as% tape, {
  out <- modelstep(fastrain())
  l <- out[1]
  acc <- out[2]
})

gradients <-
  tape$gradient(l, model$trainable_variables)
optimizer$apply_gradients(purrr::transpose(list(
  gradients, model$trainable_variables
)))


with(backend()$name_scope(optimizer2$`_name`), with(tf$python$framework$ops$init_scope(), {
    optimizer2$iterations ; optimizer2$`_create_hypers`() ; optimizer2$`_create_slots`(model$trainable_weights) }))

np = import("numpy", convert = FALSE)

wts <- backend(FALSE)$batch_get_value(optimizer$weights)
np$save("test.npy", np$array(wts, dtype = "object"), allow_pickle = TRUE)

wts2 <- np$load("test.npy", allow_pickle = TRUE)

optimizer2 <- tf$optimizers$Adam$from_config(optimizer$get_config())
optimizer2$set_weights(wts2)

optimizer$lr$assign(1)