module QMDP

using POMDPs
using GridInterpolations

import POMDPs: Solver, solve

export
    QMDPSolver,
    SampleQMDPSolver,
    solve


type QMDPSolver <: Solver
    # Functions required:
    # n_states, n_actions
    # states, actions!
    # create_action, create_transtion, create_interpolants
    # transition!, intrpolants!
    # weight, index
    max_iterations::Int64
    tolerance::Float64
    discount_factor::Float64
end


function QMDPSolver(;max_iterations::Int64=100, tolerance::Float64=1e-3, discount_factor=0.99)
    
    return QMDPSolver(max_iterations, tolerance, discount_factor)
end

function solve(solver::QMDPSolver, pomdp::POMDP; verbose::Bool=false)
    
    # state and action space info
    n_s = n_states(pomdp)
    n_a = n_actions(pomdp)
    state_iter = states(pomdp)
    action_iter = [create_action(pomdp) for i = 1:n_a]

    # solver parameters
    max_iterations = solver.max_iterations
    discount_factor = solver.discount_factor
    tolerance = solver.tolerance

    # intialize the alpha-vectors
    alphas = zeros(n_s, n_a)
    V      = zeros(n_s)

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition(pomdp)
    interps = create_interpolants(pomdp)

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        # state loop
        for (istate, s) in enumerate(state_iter)
            oldV = V[istate] # for computing residual
            actions!(action_iter, pomdp, s) 
            max_alpha = -Inf
            # action loop
            # alpha(s) = R(s,a) + discount_factor * sum(T(s'|s,a)max(alpha(s'))
            for (iaction, a) in enumerate(action_iter)
                transition!(dist, pomdp, s, a) # fills distribution over neighbors
                interpolants!(interps, dist) # fills weights and their indices
                q_new = 0.0
                for j = 1:length(interps)
                    p = weight(interps, j)
                    s_idx = index(interps, j)
                    q = V[s_idx]
                    #q = maximum(alphas[s_idx,:])
                    q_new += (p*q)
                end
                new_alpha = reward(pomdp, s, a) + discount_factor * q_new
                alphas[istate, iaction] = new_alpha 
                new_alpha > max_alpha ? (max_alpha = new_alpha) : nothing
            end # actiom
            # update the value array
            V[istate] = max_alpha 
            diff = abs(V[istate] - oldV)
            diff > residual ? (residual = diff) : nothing
        end # state
        verbose ? println("Iteration : $i, residual: $residual") : nothing
        residual < tolerance ? break : nothing 
    end # main
    return alphas
end


type SampleQMDPSolver <: Solver
    # Functions required:
    # n_states, n_actions
    # states, actions!
    # convert
    # dimensions
    # create_action, create_transtion
    # transition!
    max_iterations::Int64
    tolerance::Float64
    n_samples::Int64
    discount_factor::Float64
    cuts::Array{Array{Float64}}
end

function SampleQMDPSolver(cuts::Array{Array{Float64}};max_iterations::Int64=1000, tolerance::Float64=1e-3, n_samples::Int64=1, discount_factor::Float64=0.99)
    
    return SampleQMDPSolver(max_iterations, tolerance, n_samples, discount_factor, cuts)
end

function solve(solver::SampleQMDPSolver, pomdp::POMDP; verbose::Bool=false)
    
    # state and action space info
    n_s = n_states(pomdp)
    n_a = n_actions(pomdp)
    state_iter = states(pomdp)
    action_iter = [create_action(pomdp) for i = 1:n_a]

    max_iterations = solver.max_iterations
    n_samples      = solver.n_samples

    # Intialize the grid
    sc = solver.cuts
    grid = RectangleGrid(sc...)

    alphas = zeros(n_s, n_a)
    V      = zeros(n_s)

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition(pomdp)

    n_dims = dimensions(dist)

    sample_state = create_state(pomdp)
    sample_x = [0.0 for i = 1:n_dims]

    # TODO: pre-allocate dist and x
    for i = 1:max_iterations
        for (istate, s) in enumerate(states)
            for (iaction, a) in enumerate(actions)
                transition!(dist, pomdp, s, a) # distribution over neighbors

                q = 0.0 
                # sample from dist and update the Q values
                for _ = 1:n_samples
                    # TODO: check if x is valid state?
                    rand!(sample_state, dist)
                    convert!(x, sample_state)
                    q += interpolate(grid, V, x)
                end
                # alpha[s,a] = R(s,a) + discount_factor * sum T(s'|s,a)*V(s')
                alphas[istate, iaction] = reward(pomdp, s, a) + discount_factor * q
            end
            # update the value array
            V[istate] = max(alphas[istate,:])
        end
    end
    return alphas
end


end # module
