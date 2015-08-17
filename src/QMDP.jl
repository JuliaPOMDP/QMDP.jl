module QMDP

using POMDPs
using GridInterpolations

import POMDPs: Solver, solve!, Policy, action, value

export
    QMDPSolver,
    QMDPPolicy,
    solve!

typealias Action Any

include("vanilla.jl")

end # module
