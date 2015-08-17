# QMDP

This Julia package implements the QMDP approximate solver for POMDP/MDP planning.

## Installation

```julia
Pkg.clone("https://github.com/sisl/QMDP.jl")
```

## Usage

```julia
using QMDP
pomdp = MyPOMDP() # initialize POMDP
policy = QMDPPolicy(pomdp) # initialize the QMDP policy
solver = QMDPSolver(max_iterations=100, tolerance=1e-3)
solve!(policy, solver, pomdp, verbose=true) # key worded verbose argument for status output to console
```

To compute optimal action define a Belief with accessor functions, or use the DiscreteBelief provided in [POMDPToolbox](https://github.com/sisl/POMDPToolbox.jl).

```julia
using POMDPToolbox
ns = n_states(pomdp)
b = DiscreteBelief(ns) # initialize to a uniform belief
a = action(policy, b)
```

