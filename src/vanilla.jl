############################################################
###################### QMDP Solver #########################
############################################################

"""
    QMDPSolver
A POMDP Solver using the QMDP approximation. This algorithm assumes full observability after the first step which considers the POMDP as an MDP.
It only supports discrete states and actions POMDPs.
The internals of the QMDPSolver relies on DiscreteValueIteration.jl. You can specify the value iteration solver of your choice (sparse or regular) to the QMDPSolver constructor.
See the DiscreteValueIteration.jl documentation for more information on the parameters. By default it is using the standard `ValueIterationSolver`.

# Constructors
- `QMDPSolver(;max_iterations::Int64=100, belres::Float64=1e-3, verbose=false)` Method that uses `ValueIterationSolver` by default
- `QMDPSovler(s::Union{ValueIterationSolver, SparseValueIterationSolver})` Method that require passing a solver (e.g. `SparseValueIterationSolver`)

# Fields
- `solver::S` a `ValueIterationSolver` or `SparseValueIterationSolver`

"""
mutable struct QMDPSolver{S <: Union{ValueIterationSolver, SparseValueIterationSolver}} <: Solver
    solver::S
end
function QMDPSolver(;max_iterations::Int64=100, belres::Float64=1e-3, verbose=false)
    solver = ValueIterationSolver(max_iterations=max_iterations, belres=belres, verbose=verbose, include_Q=true)
    return QMDPSolver(solver)
end

@POMDP_require solve(solver::QMDPSolver, pomdp::POMDP) begin
    mdp = UnderlyingMDP(pomdp)
    @subreq solve(solver.solver, mdp)
end

function solve(solver::QMDPSolver, pomdp::POMDP; kwargs...)
    vi_policy = solve(solver.solver, UnderlyingMDP(pomdp))
    return AlphaVectorPolicy(pomdp, vi_policy.qmat, vi_policy.action_map)
end
