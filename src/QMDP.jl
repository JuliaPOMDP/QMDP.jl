module QMDP

using POMDPs
using POMDPModelTools
using POMDPPolicies
using DiscreteValueIteration
using POMDPLinter: @POMDP_require, @subreq

import POMDPs: Solver
import POMDPs: solve

export
    QMDPSolver,
    solve

include("vanilla.jl")

end # module
