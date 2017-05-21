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

type QMDPSolver <: Solver
    max_iterations::Int64
    tolerance::Float64
end
function QMDPSolver(;max_iterations::Int64=100, tolerance::Float64=1e-3)
    return QMDPSolver(max_iterations, tolerance)
end

type QMDPPolicy{P<:POMDP, A} <: Policy
    alphas::Matrix{Float64}
    action_map::Vector{A}
    pomdp::P
end

# constructor with an option to pass in generated alpha vectors
function QMDPPolicy(pomdp::POMDP; alphas::Matrix{Float64}=Array(Float64,0,0))
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    if !isempty(alphas)
        @assert size(alphas) == (ns,na) "Input alphas dimension mismatch"    
    else
        alphas = zeros(ns, na)
    end
    action_map = ordered_actions(pomdp)
    return QMDPPolicy(alphas, action_map, pomdp)
end

create_policy(solver::QMDPSolver, pomdp::POMDP) = QMDPPolicy(pomdp)

updater(p::QMDPPolicy) = DiscreteUpdater(p.pomdp)

@POMDP_require solve(solver::QMDPSolver, pomdp::POMDP) begin
    vi_solver = ValueIterationSolver(solver.max_iterations, solver.tolerance)
    @subreq solve(vi_solver, pomdp)
end

function solve(solver::QMDPSolver, pomdp::POMDP, policy::QMDPPolicy=create_policy(solver, pomdp); verbose::Bool=false)
    vi_solver = ValueIterationSolver(solver.max_iterations, solver.tolerance)
    vi_policy = ValueIterationPolicy(pomdp, include_Q=true)
    vi_policy = solve(vi_solver, pomdp, vi_policy, verbose=verbose)

    policy.alphas[:] = vi_policy.qmat
    return policy
end

alphas(policy::QMDPPolicy) = policy.alphas

function action(policy::QMDPPolicy, b::DiscreteBelief)
    alphas = policy.alphas
    (ns, na) = size(alphas)
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"

    util = alphas'*b.b
    ihi = indmax(util)
    return policy.action_map[ihi]
end


function value(policy::QMDPPolicy, b::DiscreteBelief)
    alphas = policy.alphas
    (ns, na) = size(alphas)
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"

    util = alphas'*b.b
    return maximum(util)
end

function value(policy::QMDPPolicy, b)
    return action(policy, DiscreteBelief(belief_vector(policy, b)))
end

function action(policy::QMDPPolicy, b)
    return action(policy, DiscreteBelief(belief_vector(policy, b)))
end

function belief_vector(policy::QMDPPolicy, b)
    bv = Array(Float64, n_states(policy.pomdp))
    for (i,s) in enumerate(ordered_states(policy.pomdp))
        bv[i] = pdf(b, s)
    end
    return bv
end
