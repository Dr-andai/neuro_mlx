# tutorial from documentation

library(bkmr, help, pos = 2, lib.loc = NULL)

set.seed(111)

data <- SimData(n = 50, M = 4)

y <- data$y
Z <- data$Z
X <- data$X

z1 <- seq(min(data$Z[, 1]), max(data$Z[, 1]), length = 20)
z2 <- seq(min(data$Z[, 2]), max(data$Z[, 2]), length = 20)

hgrid.true <- outer(z1, z2, function(x,y) apply(cbind(x,y), 1, data$HFun))

res <- persp(z1, z2, hgrid.true, theta = 30, phi = 20, expand = 0.5, 
             col = "lightblue", xlab = "", ylab = "", zlab = "")

# To fit the BKMR model, we use the kmbayes function.
# This function implements the Markov chain Monte Carlo (MCMC) algorithm.

fitkm <- kmbayes(y = y, Z = Z, X = X, iter = 10000,
verbose = FALSE, varsel = TRUE)

TracePlot(fit = fitkm, par = "beta")
