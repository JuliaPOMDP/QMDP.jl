module QMDP

using POMDPs
using POMDPModelTools
using POMDPPolicies
using DiscreteValueIteration

import POMDPs: Solver
import POMDPs: solve

export
    QMDPSolver,
    solve

include("vanilla.jl")

end # module
