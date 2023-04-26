# QMDP

[![Build Status](https://github.com/JuliaPOMDP/QMDP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/QMDP.jl/actions/workflows/CI.yml/)
[![codecov](https://codecov.io/gh/JuliaPOMDP/QMDP.jl/branch/master/graph/badge.svg?token=eh6GUxQiQg)](https://codecov.io/gh/JuliaPOMDP/QMDP.jl)

This Julia package implements the QMDP approximate solver for POMDP/MDP planning. The QMDP solver is documented in: 

* Michael Littman, Anthony Cassandra, and Leslie Kaelbling. "[Learning policies for partially observable environments: Scaling up](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.6374)." In Proceedings of the Twelfth International Conference on Machine Learning, pages 362--370, San Francisco, CA, 1995. 

## Installation

```julia
import Pkg
Pkg.add("QMDP")
```

## Usage

```julia
using QMDP
pomdp = MyPOMDP() # initialize POMDP

# initialize the solver
# key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
solver = QMDPSolver(max_iterations=20,
                    belres=1e-3,
                    verbose=true
                   ) 

# run the solver
policy = solve(solver, pomdp)
```

To compute optimal action, define a belief with the [distribution interface](http://juliapomdp.github.io/POMDPs.jl/latest/interfaces.html#Distributions-1), or use the DiscreteBelief provided in [POMDPTools](http://juliapomdp.github.io/POMDPs.jl/latest/POMDPTools/beliefs/#Implemented-Belief-Updaters).

```julia
using POMDPTools
b = uniform_belief(pomdp) # initialize to a uniform belief
a = action(policy, b)
```

In order to use the efficient `SparseValueIterationSolver` from [DiscreteValueIteration.jl](https://github.com/JuliaPOMDP/DiscreteValueIteration.jl), you can directly pass the solver to the `QMDPSolver` constructor as follows:

```julia
using QMDP, DiscreteValueIteration
pomdp = MyPOMDP()

solver = QMDPSolver(SparseValueIterationSolver(max_iterations=20, verbose=true))

policy = solve(solver, pomdp)
```
