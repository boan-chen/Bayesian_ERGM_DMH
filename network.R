library("ergm")
library("network")
library("Bergm")
library(sna)
library(igraph)
library(ggplot2)

n <- 40

g.use <- network(n, density=0.1, directed=FALSE)
random_binary_vector <- sample(c(0, 1), n, replace = TRUE)
random_binary_vector <- rbinom(n, 1, 0.3)
## print(random_binary_vector)
g.use %v% "attr" <- random_binary_vector
formula <- ~ edges + kstar(2) + triangle ## model
g.sim <- simulate(formula, nsim=1, coef=c(-3.5, -0.1, 0.5),
                  basis=g.use, control=control.simulate(
                    MCMC.burnin=1000,
                    MCMC.interval=1000))

plot(g.sim) #, vertex.cex=g.sim %v% "attr"+1)
summary(g.sim~edges+degree(0:10)+kstar(2:5)+triangle)


## 1000 Simulation
n_edge <- c(summary(g.sim~edges))
n_kstar2 <- c(summary(g.sim~kstar(2)))
n_kstar3 <- c(summary(g.sim~kstar(3)))
n_tri <- c(summary(g.sim~triangle))

# Set alpha value (transparency)
alpha_value <- 0.1
point_color <- rgb(0, 0, 1, alpha = alpha_value)
plot(n_edge, n_kstar2, main = "1000 simulated networks", xlab = "# of edge", ylab = "# of 2-star", pch = 10, col = point_color)



## estimation
model <- g.sim ~ edges + kstar(2) + triangle
## test with ergm
gest_ergm <- ergm(model)
summary(gest_ergm)

## test with bergm
gest_bergm <- bergm(model, burn.in = 500,
               main.iters = 3000, aux.iters = 2500, nchains = 8, gamma = 0.6)
summary(gest_bergm)

## save the network
# Specify the full path to the download location and filename
download_location <- "C:/Users/HarryHuang/Desktop/112-1_bayesian_statistic_method/"
filename <- "doc_save_test_1.csv"
full_path <- paste0(download_location, filename) # Create the full file path

write.csv(
  x         = as.matrix(g.sim),
  file      = full_path,
  row.names = FALSE
)

## Goodness-of-fit
gof_ergm <- gof(gest_ergm)
# Plot all three on the same page with nice margins
par(mfrow=c(2,3))
par(oma=c(0.5,2,1,0.5))
plot(gof_ergm)

gof_bergm <- bgof(gest_bergm, aux.iters = 5000, n.deg = 15, n.dist = 9, n.esp = 8)
par(mfrow=c(2,3))
par(oma=c(0.5,2,1,0.5))
plot(gof_bergm)


## Assessing MCMC diagnostics
mcmc.diagnostics(gest_ergm)
plot(gest_bergm, lag = 100)

