### get optimal length for stride
stridecalc <- function(len, plen) {
  vec <- c()
  for (i in ceiling(plen / 3):(floor(plen / 2) - 1)) {
    if ((len - plen) %% i == 0) {
      vec <- c(vec, i)
    }
  }
  return(vec)
}

stridecalc(8100, 500)
stridecalc(6700, 500)
stridecalc(7000, 1000)
stridecalc(7600, 2000)
max(stridecalc(8100, 500))

nopatchescalc <- function(plen, maxlen, stride) {
  #stride <- plen * 0.4
  ((maxlen - plen)/stride) + 1
}

maxlencalc <- function(plen, nopatches, stride) {
  #stride <- plen * 0.4
  (nopatches - 1) * stride + plen
}

maxlencalc(500, 32, 200)
maxlencalc(1000, 16, 400)
maxlencalc(2000, 8, 800)

enckz <- function(ML, PL, NP) {
  library(numbers)
  divi <- head(divisors(PL)[-1], -1)
  
  for (i in divi) {
    C1 <- (((ML - (PL / i)) / (PL * 0.1)) + 1)
    C2 <- floor(((C1 - i) / 2) + 1)
    if (floor(C2 / 2) == NP) {
      out <- i
    }
  }
  return(out)
}

sgdr <- function(lrmin = 5e-10,
  lrmax = 5e-2,
  restart = 50,
  mult = 1,
  epoch = NULL) {
  iter <- c()
  position <- c()
  i <- 0
  while (length(iter) < epoch) {
    iter <- c(iter, rep(i, restart * mult ^ i))
    position <- c(position, c(1:(restart * mult ^ i)))
    i <- i + 1
  }
  restart2 <- (restart * mult ^ iter[epoch])
  epoch <- position[epoch]
  return(lrmin + 1 / 2 * (lrmax - lrmin) * (1 + cos((epoch / restart2) * pi)))
}

stepdecay <- function(lrmax = 0.005,
  newstep = 50,
  mult = 0.7,
  epoch = NULL) {
  return(lrmax * (mult ^ (floor((epoch
  ) / newstep))))
}

exp_decay <- function(lrmax = 0.005, mult = 0.1, epoch = NULL){
  return(lrmax * exp(-mult*epoch))
}