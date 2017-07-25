
push!(LOAD_PATH, "/home/tim/.julia/v0.3/POMDPs/pomdps.jl/src/") # TODO(tim): remove this once POMDPs is actually on path

# Problem Definition
####################################

# S ∈ {s₁ ∈ R¹, s₂ ∈ N¹, isdead::Bool}
# A ∈ {a₁ ∈ R¹, a₂ ∈ N¹}
# T(s,a) → (N(s₁ + a₁, |a₁|+0.1), s₂ + a₂ + {-1,0,1}) 90% of the time
# R(s,a) → dot([s₁², s₂, a₁, a₂], reward_weights)

using POMDPs
using Distributions
using Discretizers
using POMDPToolbox
using QMDP

import POMDPs.states
import POMDPs.actions!
import POMDPs.create_action
import POMDPs.create_transition
import POMDPs.transition!
import POMDPs.reward
import POMDPs.n_states
import POMDPs.n_actions

import POMDPToolbox: interpolants!

mutable struct MyPOMDP <: POMDP
    r_sa::Float64
    r_sb::Float64
    r_aa::Float64
    r_ab::Float64
    r_dead::Float64
    prob_random_death::Float64
end

mutable struct MyState
    a::Float64
    b::Int
    isdead::Bool
end

mutable struct MyAction
    a::Float64
    b::Int
end

mutable struct MyDistribution <: AbstractDistribution
    isdead::Bernoulli
    realpart::Normal
    discpart::Categorical
    disc_centered::Int

    MyDistribution() = new(Bernoulli(0.5), Normal(), Categorical([0.25,0.5,0.25]), 0)
end

create_state(::MyPOMDP) = MyState(0.0, 0, false)
create_action(::MyPOMDP) = MyAction(0.0, 0)
create_transition(::MyPOMDP) = MyDistribution()

function transition!(d::MyDistribution, p::MyPOMDP, s::MyState, a::MyAction)
    if s.isdead
        # d.isdead.p = 1.0
        d.isdead = Bernoulli(1.0)
    else
        # d.isdead.p = p.prob_random_death
        # d.realpart.μ = s.a + a.a
        # d.realpart.σ = abs(a.a) + 0.1
        d.isdead = Bernoulli(p.prob_random_death)
        d.realpart = Normal(s.a + a.a, abs(a.a) + 0.1)
        d.disc_centered = s.b + a.b
    end

    d
end
function reward(p::MyPOMDP, s::MyState, a::MyAction)
    if s.isdead
        p.r_dead
    else
        s.a * p.r_sa + s.b * p.r_sb + a.a * p.r_aa + a.b * p.r_ab
    end
end

function rand!(s::MyState, d::MyDistribution)
    if isapprox(rand(d.isdead), 1.0)
        s.isdead = true
    else
        s.a = rand(d.realpart)
        s.b = rand(d.discpart) - 2 + d.disc_centered
    end
    s
end

# Definitions for Solver
####################################

mutable struct DiscretizationParams
    real::Vector{Float64}
    disc::Vector{Int}
end

my_disc_params = DiscretizationParams(linspace(-10.0,10.0,10),[-10:10])

const ACTION_SET_A = [MyAction(-1.0,0), MyAction(1.0,0), MyAction(0.0,-1), MyAction(0.0,1)]
const ACTION_SET_B = [MyAction(-0.5,0), MyAction(0.5,0), MyAction(0.0,-2), MyAction(0.0,2)]

get_ndiscrete_states(params::DiscretizationParams) = (length(params.real)-1) * (length(params.disc)-1) + 1

function index2state(i::Int, params::DiscretizationParams=my_disc_params)
    if i == get_ndiscrete_states(params)
        MyState(0.0, 0, true)
    else
        bin_real,bin_disc = ind2sub((length(params.real)-1, length(params.disc)-1), i)
        real = params.real[bin_real]
        disc = params.disc[bin_disc]
        MyState(real, disc, false)
    end
end
# function state2index(s::MyState, params::DiscretizationParams=my_disc_params)
#     if s.isdead
#         get_ndiscrete_states(params)
#     else
#         bin_real = encode(params.real, s.a)
#         bin_disc = encode(params.disc, s.b)
#         sub2ind((nlabels(params.real), nlabels(params.disc)), bin_real, bin_disc)
#     end
# end

n_states(pomdp::MyPOMDP) = length(states(pomdp))
n_actions(pomdp::MyPOMDP) = length(ACTION_SET_A)

function states(::MyPOMDP, params::DiscretizationParams=my_disc_params)
    nstates = get_ndiscrete_states(params)
    map(i->index2state(i), 1:nstates)
    # 1:nstates
end
function actions!(retval::Vector{MyAction}, ::MyPOMDP, s::MyState, params::DiscretizationParams=my_disc_params)
    if s.a > 0.0
        retval[1:end] = ACTION_SET_A[1:end]
    else
        retval[1:end] = ACTION_SET_B[1:end]
    end
    retval
end

##############


function interpolants!(interpolants::Interpolants, d::MyDistribution;
    params::DiscretizationParams=my_disc_params
    )

     # we can be dead
     # we can have 3 different discrete components
     # with a full set of continuous components

     empty!(interpolants)

    if isapprox(d.isdead.p, 1.0)
        push!(interpolants, get_ndiscrete_states(params), 1.0) # [(STATE_DEAD,1.0)]
    else
        weights_real = Array{Float64}(length(params.real))
        weights_disc = d.discpart.p

        weights_real[1] = cdf(d.realpart, (params.real[2]+params.real[1])/2)
        for i = 2 : length(params.real)-2
             weights_real[i] = cdf(d.realpart, (params.real[i+1]+params.real[i])/2) - cdf(d.realpart, (params.real[i]+params.real[i-1])/2)
        end
        weights_real[end] = 1.0 - cdf(d.realpart, (params.real[end]+params.real[end-1])/2)

        prob_not_dead = 1.0 - d.isdead.p
        for i = 1 : length(weights_real)
            for j = 1 : length(weights_disc)
                #println(count,  ") ", i, "  ", j, ": ", weights_real[i], "  ", weights_disc[j], "  ", prob_not_dead)
                prob = weights_real[i] * weights_disc[j] * prob_not_dead
                state_index = sub2ind((length(params.real)-1, length(params.disc)-1), i, j)
                push!(interpolants, state_index, prob)
            end
        end

        push!(interpolants, get_ndiscrete_states(params), d.isdead.p)
    end

    interpolants
end

# Problem Usage
####################################

pomdp = MyPOMDP(0.5,0.5,-1.0,-1.0,-100.0,0.1)

#solver = QMDP(max_iterations=1)

#alphas = solve(solver, pomdp)
