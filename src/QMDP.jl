module QMDP

using POMDPs
using POMDPToolbox

import POMDPs: Solver, Policy
import POMDPs: create_policy, solve, action, value, update, initialize_belief, updater, create_belief, create_policy

export
    QMDPSolver,
    QMDPPolicy,
    QMDPUpdater,
    solve,
    action,
    value,
    update,
    initialize_belief,
    updater,
    create_policy,
    create_belief

include("vanilla.jl")

end # module
