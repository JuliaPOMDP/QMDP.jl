############################################################
###################### QMDP Solver #########################
############################################################

#=
POMDP Model Requirements:
    create_transition_distribution(pomdp)
    states(pomdp)
    actions(pomdp)
    iterator(space)
    transition(pomdp, s, a, dist)
    reward(pomdp, s, a, sp)
    pdf(dist, sp)
    discount(pomdp)
    state_index(pomdp, sp)

    The user must implement the above functions to use QMDP.
=#

mutable struct QMDPSolver <: Solver
    max_iterations::Int64
    tolerance::Float64
end
function QMDPSolver(;max_iterations::Int64=100, tolerance::Float64=1e-3)
    return QMDPSolver(max_iterations, tolerance)
end

@POMDP_require solve(solver::QMDPSolver, pomdp::POMDP) begin
    vi_solver = ValueIterationSolver(solver.max_iterations, solver.tolerance)
    @subreq solve(vi_solver, pomdp)
end

function solve(solver::QMDPSolver, pomdp::POMDP; verbose::Bool=false)
    vi_solver = ValueIterationSolver(solver.max_iterations, solver.tolerance)
    vi_policy = ValueIterationPolicy(pomdp, include_Q=true)
    vi_policy = solve(vi_solver, pomdp, vi_policy, verbose=verbose)

    return AlphaVectorPolicy(pomdp, vi_policy.qmat)
end
