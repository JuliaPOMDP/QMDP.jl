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
    verbose::Bool
end
function QMDPSolver(;max_iterations::Int64=100, tolerance::Float64=1e-3, verbose=false)
    return QMDPSolver(max_iterations, tolerance, verbose)
end

@POMDP_require solve(solver::QMDPSolver, pomdp::POMDP) begin
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, belres=solver.tolerance)
    @subreq solve(vi_solver, pomdp)
end

function solve(solver::QMDPSolver, pomdp::POMDP; kwargs...)
    # deprecation warning - can be removed when Julia 1.0 is adopted
    if !isempty(kwargs)
        @warn "Keyword args for solve(::QMDPSolver, ::POMDP) are no longer supported. For verbose output, use the verbose option in the ValueIterationSolver"
    end
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, belres=solver.tolerance, verbose=solver.verbose, include_Q=true)
    vi_policy = solve(vi_solver, pomdp)

    return AlphaVectorPolicy(pomdp, vi_policy.qmat)
end
