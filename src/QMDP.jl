module QMDP

using POMDPs
using POMDPTools
using DiscreteValueIteration
using POMDPLinter: @POMDP_require, @subreq

import POMDPs: Solver
import POMDPs: solve

export
    QMDPSolver,
    solve

include("vanilla.jl")

end # module
