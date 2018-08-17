module QMDP

using POMDPs
using DiscreteValueIteration
using POMDPPolicies
import POMDPs: Solver
import POMDPs: solve

export
    QMDPSolver,
    solve

include("vanilla.jl")

end # module
