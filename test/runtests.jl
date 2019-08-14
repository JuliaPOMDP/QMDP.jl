using QMDP
using POMDPs
using POMDPModels
using POMDPModelTools
using DiscreteValueIteration
using BeliefUpdaters
using POMDPTesting
using Test
using Random


@testset "Standard QMDP" begin
    pomdp = TigerPOMDP()
    solver = QMDPSolver(verbose=true)

    @requirements_info solver pomdp

    policy = solve(solver, pomdp)

    solver = QMDPSolver(verbose=false)
    rng = MersenneTwister(11)

    bu = updater(policy)
    sd = initialstate_distribution(pomdp)
    b = initialize_belief(bu, sd)

    a = action(policy, b)
    v = value(policy, b)

    s = initialstate(pomdp, rng)
    sp, o = generate_so(pomdp, s, a, rng)
    bp = update(bu, b, a, o)
    @test isa(bp, DiscreteBelief)

    r = test_solver(solver, pomdp)
    @test isapprox(r, 17.711, atol=1e-2)
end

@testset "Sparse QMDP" begin
    pomdp = TigerPOMDP()
    solver = QMDPSolver(SparseValueIterationSolver())

    r = test_solver(solver, pomdp)
    @test isapprox(r, 17.711, atol=1e-2)
end