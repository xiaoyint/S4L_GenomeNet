args = commandArgs(trailingOnly = TRUE)
mod    <- args[1] # "all_lm", "bac_rn18_16"
size   <- as.integer(args[2])
perp   <- as.integer(args[3])
if (is.na(size)) {
  size <- 10000
  perp <- 3000
}

packages_required <-
  c("ggplot2", "Rtsne", "viridis", "data.table", "dplyr")
not_installed <-
  packages_required[!packages_required %in% installed.packages()[, "Package"]]
if (length(not_installed) > 0) {
  lapply(not_installed, install.packages)
}
lapply(packages_required, library, character.only = TRUE)

purple <- "#332288"
blue <- "#88CCEE"
pink <- "#AA4499"

source("R/help/getenvBacDive.R")
reps <-
  readRDS(paste0(preloadGeneratorpath, "states/gram/", "test_", mod, ".rds"))

lreps <- list()
for (i in levels(reps$target)) {
  lreps[[i]] <- reps[target == i,]
  lreps[[i]] <-
    lreps[[i]][sample(1:nrow(lreps[[i]]), ceiling(size / length(levels(reps$target)))), ]
}
dfreps <- rbindlist(lreps)

tsneplot <- function(perplexity) {
  tsne_results <-
    Rtsne(dfreps[, 1:(length(dfreps) - 1)][, names(dfreps[, 1:(length(dfreps) - 1)]) := lapply(.SD, as.numeric)],
      perplexity = perplexity,
      check_duplicates = FALSE)
  
  tsne_plot <-
    data.frame(x = tsne_results$Y[, 1],
      y = tsne_results$Y[, 2],
      col = dfreps$target)
  
  saveRDS(tsne_plot,
    paste0("tsne_results/gram_", mod, "_", perplexity, ".rds"))
  
  ggplot(tsne_plot, aes(x, y, color = col)) + geom_point() +
    theme_minimal() + scale_y_continuous(name = "") +
    scale_x_continuous(name = "") +
    scale_color_manual(
      name = "Gram Stain",
      values = c(purple, blue, pink),
      labels = c("negative", "positive")
    ) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_blank()
    )
}

cat(format(Sys.time(), "%F %R"), ": Start plot creation\n")
png(
  filename = paste("tsne_results/gram", mod, perp, ".png", sep = "_"),
  width = 750,
  height = 500,
  units = "px",
  res = 95
)
tsneplot(30000)
dev.off()
cat(format(Sys.time(), "%F %R : "), perp, "done\n")