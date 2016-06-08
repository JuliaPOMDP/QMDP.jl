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
        am = Any[]
        space = actions(pomdp)
        for a in iterator(space)
            push!(am, a)
        end
        self.action_map = am
        self.pomdp = pomdp
        return self
    end
end

create_policy(solver::QMDPSolver, pomdp::POMDP) = QMDPPolicy(pomdp)

updater(p::QMDPPolicy) = DiscreteUpdater(p.pomdp)

function solve(solver::QMDPSolver, pomdp::POMDP, policy::QMDPPolicy=create_policy(solver, pomdp); verbose::Bool=false)

    # solver parameters
    max_iterations = solver.max_iterations
    tolerance = solver.tolerance
    discount_factor = discount(pomdp)

    # intialize the alpha-vectors
    alphas = policy.alphas

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition_distribution(pomdp)

    # initalize space
    sspace = states(pomdp)
    aspace = actions(pomdp)

    total_time = 0.0
    iter_time = 0.0

    # main loop
    for i = 1:max_iterations
        tic()
        residual = 0.0
        # state loop
        for (istate, s) in enumerate(iterator(sspace))
            old_alpha = maximum(alphas[istate,:]) # for residual 
            max_alpha = -Inf
            # action loop
            # alpha(s) = R(s,a) + discount_factor * sum(T(s'|s,a)max(alpha(s'))
            for (iaction, a) in enumerate(iterator(aspace))
                dist = transition(pomdp, s, a, dist) # fills distribution over neighbors
                q_new = 0.0
                for sp in iterator(sspace)
                    p = pdf(dist, sp)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(pomdp, s, a, sp)
                    sidx = state_index(pomdp, sp)
                    q_new += p * (r + discount_factor * maximum(alphas[sidx,:]))
                end
                new_alpha = q_new
                alphas[istate, iaction] = new_alpha
                new_alpha > max_alpha ? (max_alpha = new_alpha) : nothing
            end # actiom
            # update the value array
            diff = abs(max_alpha - old_alpha)
            diff > residual ? (residual = diff) : nothing
        end # state
        iter_time = toq()
        total_time += iter_time
        verbose ? println("Iteration : $i, residual: $residual, iteration run-time: $iter_time, total run-time: $total_time") : nothing
        residual < tolerance ? break : nothing 
    end # main
    policy
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
