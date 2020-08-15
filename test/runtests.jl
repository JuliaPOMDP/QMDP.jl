using QMDP
using POMDPs
using POMDPModels
using POMDPModelTools
using DiscreteValueIteration
using BeliefUpdaters
using POMDPTesting
using POMDPLinter: show_requirements, get_requirements
using Test
using Random


@testset "Standard QMDP" begin
    pomdp = TigerPOMDP()
    solver = QMDPSolver(verbose=true)

    @test_skip @requirements_info solver pomdp
    show_requirements(get_requirements(POMDPs.solve, (solver, pomdp)))

    policy = solve(solver, pomdp)

    solver = QMDPSolver(verbose=false)
    rng = MersenneTwister(11)

    bu = updater(policy)
    sd = initialstate(pomdp)
    b = initialize_belief(bu, sd)

    a = action(policy, b)
    v = value(policy, b)

    s = rand(rng, initialstate(pomdp))
    sp, o = @gen(:sp, :o)(pomdp, s, a, rng)
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
