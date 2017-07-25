using QMDP
using POMDPs
using POMDPModels
using POMDPToolbox
using Base.Test

pomdp = TigerPOMDP()
solver = QMDPSolver()

@requirements_info solver pomdp

policy = solve(solver, pomdp, verbose=true)
rng = MersenneTwister(11)

bu = updater(policy)
sd = initial_state_distribution(pomdp)
b = create_belief(bu)
b = initialize_belief(bu, sd, b)

a = action(policy, b)
v = value(policy, b)

s = initial_state(pomdp, rng)
sp, o = generate_so(pomdp, s, a, rng)
bp = update(bu, b, a, o)
@test isa(bp, DiscreteBelief)

r = test_solver(solver, pomdp)

@test isapprox(r, 17.711, atol=1e-2)
