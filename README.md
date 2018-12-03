# QMDP

[![Build Status](https://travis-ci.org/JuliaPOMDP/QMDP.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/QMDP.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/QMDP.jl/badge.svg)](https://coveralls.io/r/JuliaPOMDP/QMDP.jl)

This Julia package implements the QMDP approximate solver for POMDP/MDP planning. The QMDP solver is documented in: 

* Michael Littman, Anthony Cassandra, and Leslie Kaelbling. "[Learning policies for partially observable environments: Scaling up](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.6374)." In Proceedings of the Twelfth International Conference on Machine Learning, pages 362--370, San Francisco, CA, 1995. 

## Installation

```julia
using POMDPs, Pkg
POMDPs.add_registry()
Pkg.add("QMDP")
```

## Usage

```julia
using QMDP
pomdp = MyPOMDP() # initialize POMDP

# initialize the solver
# key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
solver = QMDPSolver(max_iterations=20,
                    tolerance=1e-3,
                    verbose=true
                   ) 

# run the solver
policy = solve(solver, pomdp)
```

To compute optimal action, define a belief with the [distribution interface](http://juliapomdp.github.io/POMDPs.jl/latest/interfaces.html#Distributions-1), or use the DiscreteBelief provided in [BeliefUpdaters](https://github.com/JuliaPOMDP/BeliefUpdaters.jl).

```julia
using BeliefUpdaters
b = uniform_belief(pomdp) # initialize to a uniform belief
a = action(policy, b)
```
