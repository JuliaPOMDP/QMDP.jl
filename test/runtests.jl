using QMDP
using POMDPModels
using Base.Test

pomdp = TigerPOMDP()
solver = QMDPSolver()
policy = solve(solver, pomdp, verbose=true)

bu = updater(policy)
sd = initial_state_distribution(pomdp)
b = create_belief(bu)
b = initialize_belief(bu, sd, b)

a = action(policy, b)
v = value(policy, b)
