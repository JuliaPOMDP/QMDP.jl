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

type QMDPPolicy <: Policy
    alphas::Matrix{Float64}
    action_map::Vector{Any}
    pomdp::POMDP
    # constructor with an option to pass in generated alpha vectors
    function QMDPPolicy(pomdp::POMDP; alphas::Matrix{Float64}=Array(Float64,0,0))
        ns = n_states(pomdp)
        na = n_actions(pomdp)
        self = new()
        if !isempty(alphas)
            @assert size(alphas) == (ns,na) "Input alphas dimension mismatch"    
            self.alphas = alphas
        else
            self.alphas = zeros(ns, na)
        end
        self.action_map = ordered_actions(pomdp)
        self.pomdp = pomdp
        return self
    end
end

create_policy(solver::QMDPSolver, pomdp::POMDP) = QMDPPolicy(pomdp)

updater(p::QMDPPolicy) = DiscreteUpdater(p.pomdp)

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
    ihi = 0
    vhi = -Inf
    (ns, na) = size(alphas)
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"
    for ai = 1:na
        util = dot(alphas[:,ai], b.b)
        if util > vhi
            vhi = util
            ihi = ai
        end
    end
    return policy.action_map[ihi]
end

function value(policy::QMDPPolicy, b::DiscreteBelief)
    alphas = policy.alphas
    vhi = -Inf
    (ns, na) = size(alphas)
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"
    for ai = 1:na
        util = 0.0
        for si = 1:length(b.b)
            util += b.b[si] * alphas[si,ai]
        end
        if util > vhi
            vhi = util
        end
    end
    return vhi
end
