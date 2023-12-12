# Bayesian Estimation for Exponential Random Graph Models (ERGM)

## Background
This code implements a Bayesian estimation approach for exponential random graph models (ERGMs) using a double Metropolis-Hasting (DMH) algorithm. ERGMs are statistical models used for modeling social network formation and structure. However, they face challenges with intractable likelihoods. This code explores using a DMH algorithm to sample from the posterior distribution.  The core algorithm structure involves:

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

## Key files:

In **presentation**, we explore the models and issues that arose from our simulation results. We also make comments on potential improvement by reviewing essential literature.

In **python_ver3**, we provide:

1. `DMH.py`: Implements the DMH algorithm for ERGM posterior sampling
2. `DGP.py`: Contains functions to generate ERGM networks
3. `estimation_r.py`: Runs estimation on networks simulated from R
4. `estimation_self.py`: Runs estimation on networks simulated from our Python code
5. `network_analysis.py`: Analyzes properties of simulated networks

## Usage
The main files for running estimation are `estimation_r.py` and `estimation_self.py`.

`estimation_r.py` takes an adjacency matrix simulated from R's `ergm` package, runs multiple DMH chains in parallel, and collects the sampling results.

`estimation_self.py` first simulates a network using our network_metropolis generator, visualizes the network, and then runs multiple DMH chains and estimation. Note that our model is designed only for estimating the effects of edges, two-stars, and triangles. The number of parallel chains, burn-in, and main sampling iterations can be configured within these files. Trace plots and posterior sample collection are automated.

`network_analysis.py` can be used independently to simulate multiple networks for a given set of ERGM parameters and analyze the distribution of network statistics. This is useful for diagnosing mixing or degeneracy issues.





