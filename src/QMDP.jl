module QMDP

using POMDPs

import POMDPs: Solver, solve, create_policy, Policy, action, value

export
    QMDPSolver,
    QMDPPolicy,
    solve,
    action,
    value,
    create_policy

include("vanilla.jl")

end # module
