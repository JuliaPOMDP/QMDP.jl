using QMDP
using POMDPs
using POMDPModels
using POMDPTools
using DiscreteValueIteration
using POMDPLinter: show_requirements, get_requirements
using Test
using Random


@testset "Standard QMDP" begin
    pomdp = TigerPOMDP()
    solver = QMDPSolver(verbose=true)

    @test_skip @requirements_info solver pomdp
    show_requirements(get_requirements(POMDPs.solve, (solver, pomdp)))

    policy = solve(solver, pomdp)
    
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

    # Functional test for solver
    solver = QMDPSolver(verbose=false)
    test_solver(solver, pomdp)
    
    solver = QMDPSolver(; verbose=false, max_iterations=1000, belres=1e-10)
    policy = solve(solver, pomdp)
    @test length(actions(pomdp)) == length(policy.alphas)
    
    known_left_or_right = pomdp.r_escapetiger / (1 - discount(pomdp))
    # assuming 0.5 * pomdp.r_findtiger + 0.5 * pomdp.r_escapetiger < r_listen 
    uniform_value = pomdp.r_listen + discount(pomdp) * known_left_or_right
    
    @test isapprox(value(policy, BoolDistribution(1.0)), known_left_or_right, atol=1e-3)
    @test isapprox(value(policy, BoolDistribution(0.0)), known_left_or_right, atol=1e-3)
    @test isapprox(value(policy, BoolDistribution(0.5)), uniform_value, atol=1e-3)
end

@testset "Sparse QMDP" begin
    pomdp = TigerPOMDP()
    solver = QMDPSolver(SparseValueIterationSolver())

    # Functional test for solver
    test_solver(solver, pomdp)
    solver = QMDPSolver(SparseValueIterationSolver(; verbose=false, max_iterations=1000, belres=1e-10))
    policy = solve(solver, pomdp)
    @test length(actions(pomdp)) == length(policy.alphas)
    
    known_left_or_right = pomdp.r_escapetiger / (1 - discount(pomdp))
    # assuming 0.5 * pomdp.r_findtiger + 0.5 * pomdp.r_escapetiger < r_listen 
    uniform_value = pomdp.r_listen + discount(pomdp) * known_left_or_right
    @test isapprox(value(policy, BoolDistribution(1.0)), known_left_or_right, atol=1e-3)
    @test isapprox(value(policy, BoolDistribution(0.0)), known_left_or_right, atol=1e-3)
    @test isapprox(value(policy, BoolDistribution(0.5)), uniform_value, atol=1e-3)
end
