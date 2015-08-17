module QMDP

using POMDPs

import POMDPs: Solver, solve!, Policy, action, value

export
    QMDPSolver,
    QMDPPolicy,
    solve!,
    action,
    value

typealias Action Any

include("vanilla.jl")

end # module
