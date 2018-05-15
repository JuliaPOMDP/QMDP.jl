module QMDP

using POMDPs
using POMDPToolbox
using DiscreteValueIteration

import POMDPs: Solver
import POMDPs: solve

export
    QMDPSolver,
    solve,
    create_policy

include("vanilla.jl")

end # module
