using QMDP
using POMDPModels
using Base.Test

pomdp = TigerPOMDP()
solver = QMDPSolver()
policy = solve(solver, pomdp, verbose=true)
