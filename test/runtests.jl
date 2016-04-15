using QMDP
using POMDPModels
using Base.Test

# write your own tests here
@test 1 == 1

pomdp = TigerPOMDP()
solver = QMDPSolver()
policy = solve(solver, pomdp, verbose=true)
