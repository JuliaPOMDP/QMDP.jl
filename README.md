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
solver = QMDPSolver(max_iterations=100, tolerance=1e-3, discount_factor=0.99)
alphas = solve(solver, pomdp)
```
