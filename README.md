# QMDP

[![Build Status](https://travis-ci.org/JuliaPOMDP/QMDP.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/QMDP.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/QMDP.jl/badge.svg)](https://coveralls.io/r/JuliaPOMDP/QMDP.jl)

This Julia package implements the QMDP approximate solver for POMDP/MDP planning. The QMDP solver is documented in: 

* Michael Littman, Anthony Cassandra, and Leslie Kaelbling. [Learning policies for partially observable environments: Scaling up. In Armand Prieditis and Stuart Russell](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.6374), editors, Proceedings of the Twelfth International Conference on Machine Learning, pages 362--370, San Francisco, CA, 1995. Morgan Kaufmann. 

## Installation

```julia
Pkg.clone("https://github.com/sisl/QMDP.jl")
```

## Usage

```julia
using QMDP
pomdp = MyPOMDP() # initialize POMDP

# initialize the solver
# key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
solver = QMDPSolver(max_iterations=20, tolerance=1e-3) 

# initialize the QMDP policy
policy = create_policy(solver, pomdp)

# run the solver
solve(solver, pomdp, policy, verbose=true)
```

To compute optimal action define a Belief with accessor functions, or use the DiscreteBelief provided in [POMDPToolbox](https://github.com/sisl/POMDPToolbox.jl).

```julia
using POMDPToolbox
ns = n_states(pomdp)
b = DiscreteBelief(ns) # initialize to a uniform belief
a = action(policy, b)
```

