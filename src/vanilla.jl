type QMDPSolver <: Solver
    # Functions required:
    # n_states, n_actions
    # states, actions!
    # create_action, create_transtion, create_interpolants
    # transition!, intrpolants!
    # weight, index
    max_iterations::Int64
    tolerance::Float64
end
function QMDPSolver(;max_iterations::Int64=100, tolerance::Float64=1e-3)
    return QMDPSolver(max_iterations, tolerance)
end

type QMDPPolicy <: Policy
    alphas::Matrix{Float64}
    action_map::Vector{Action}
    # constructor with an option to pass in generated alpha vectors
    function QMDPPolicy(pomdp::POMDP; alphas::Matrix{Float64}=Array(Float64,0,0))
        ns = n_states(pomdp)
        na = n_actions(pomdp)
        self = new()
        if !isempty(alphas)
            @assert size(alphas,1) == ns && size(alphas,2) == na "Input alphas dimension mismatch"    
            self.alphas = alphas
        else
            self.alphas = zeros(ns, na)
        end
        am = Action[]
        space = actions(pomdp)
        for a in domain(space)
            push!(am, a)
        end
        self.action_map = am
        return self
    end
end


function solve!(policy::QMDPPolicy, solver::QMDPSolver, pomdp::POMDP; verbose::Bool=false)

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
        for (istate, s) in enumerate(domain(sspace))
            old_alpha = maximum(alphas[istate,:]) # for residual 
            actions!(aspace, pomdp, s) 
            max_alpha = -Inf
            # action loop
            # alpha(s) = R(s,a) + discount_factor * sum(T(s'|s,a)max(alpha(s'))
            for (iaction, a) in enumerate(domain(aspace))
                transition!(dist, pomdp, s, a) # fills distribution over neighbors
                q_new = 0.0
                for j = 1:length(dist)
                    p = weight(dist, j)
                    sidx = index(dist, j)
                    q_new += p * maximum(alphas[sidx,:])
                end
                new_alpha = reward(pomdp, s, a) + discount_factor * q_new
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

function action(policy::QMDPPolicy, b::Belief)
    alphas = policy.alphas
    ihi = 0
    vhi = -Inf
    (ns, na) = size(alphas)
    @assert length(b) == ns "Length of belief and alpha-vector size mismatch"
    for ai = 1:na
        util = 0.0
        for si = 1:length(b)
            util += weight(b,si) * alphas[si,ai]
        end
        if util > vhi
            vhi = util
            ihi = ai
        end
    end
    return policy.action_map[ihi]
end


