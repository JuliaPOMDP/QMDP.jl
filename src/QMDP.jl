module QMDP

using POMDPs
using POMDPToolbox
using DiscreteValueIteration

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater

export
    QMDPSolver,
    QMDPPolicy,
    QMDPUpdater,
    solve,
    action,
    value,
    state_value,
    update,
    initialize_belief,
    updater,
    create_policy,
    create_belief

include("vanilla.jl")

end # module
