# Bayesian Estimation for Exponential Random Graph Model (ERGM)

## Introduction

The estimation of the posterior for Exponential Random Graph Models (ERGM) often faces intractability, requiring sophisticated methods such as Markov Chain Monte Carlo (MCMC) estimation (Snijder, 2002) or alternative approaches. Our Bayesian estimation approach for ERGM utilizes the double Metropolis-Hastings (DMH) algorithm first proposed by Liang (2010). This algorithm introduces auxiliary states to address the intractable nature of likelihood estimation. The core algorithm structure involves:

```
for t = 1 to T:
    Generate θ′ ∼ h(·|θ)
    Sample an auxiliary variable y′ ∼ π(y′|θ′) using an exact sampler
    Compute r(θ, θ′, y′|y) = π(y|θ′)π(θ′) / (π(y′|θ)π(θ)) * h(θ|θ′, y) / (π(y|θ)π(θ)) * π(y′|θ′)h(θ′|θ,y)
    Draw u ∼ Uniform(0, 1)
    if u < r then set θ = θ′
end for
```
The normalizing constant z(θ) is canceled by introducing π(y′|θ).

## Code Overview
### Python
The Python code encompasses both data generation and the estimation program for the DMH algorithm. The programs are structured by python Class. The current implementation utilizes adaptive Monte Carlo for the Metropolis-Hastings (MH) algorithm. Future iterations may explore Hamiltonian Monte Carlo coupled with simulated annealing to enhance the acceptance rate and convergence. The code architecture primarily draws inspiration from a MATLAB code provided by Prof. Hsieh, Department of Economics.

### R
The R code is an evolving program aimed at optimizing the use of rstan within our project. [Stan](https://mc-stan.org/) is a package optimized for Bayesian statistical analysis. Implementing Stan could enhance multiple MH processes within the algorithm and offer improved tracking of the implementation. Although Stan is also available for PyStan, our current limitations with access lead us to focus on R for development purposes.

## Data Generating Process (DGP) Results
The DGP results are based on a beta parameterization (-3, 1, 1, -1) using our Python-based DGP code. Results include a histogram of edges and a density plot depicting the number of edges and the maximum degree generated.

![image](https://github.com/boan-chen/Bayesian_ERGM/assets/108161781/90469e47-890f-4474-b465-f82bb4625f00)
![image](https://github.com/boan-chen/Bayesian_ERGM/assets/108161781/961b4932-223e-4841-b7e6-59a60ba46cdc)

## References
1. Caimo, A., Bouranis, L., Krause, R., & Friel, N. (2022). Statistical Network Analysis with Bergm. Journal of Statistical Software, 104(1), 1–23. https://doi.org/10.18637/jss.v104.i01
2. Snijders, T.A. (2002). Markov Chain Monte Carlo Estimation of Exponential Random Graph Models. J. Soc. Struct., 3.
3. Liang, Faming. (2010). A double Metropolis–Hastings sampler for spatial models with intractable normalizing constants. Journal of Statistical Computation and Simulation. 80. 1007-1022.
4. Hermans, J., Begy, V. &amp; Louppe, G.. (2020). Likelihood-free MCMC with Amortized Approximate Ratio Estimators. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:4239-4248 Available from https://proceedings.mlr.press/v119/hermans20a.html.
5. Salazar, R., Toral, R. Simulated Annealing Using Hybrid Monte Carlo. J Stat Phys 89, 1047–1060 (1997). https://doi.org/10.1007/BF02764221
6. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv, arXiv:1701.02434 [stat.ME]. https://doi.org/10.48550/arXiv.1701.02434





