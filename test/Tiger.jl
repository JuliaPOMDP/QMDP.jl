#module Tiger

### Don't run Tiger with QMDP! QMDP can't handle info-gathering actions.

using POMDPs
using Distributions
using POMDPToolbox

import POMDPs.states
import POMDPs.actions!
import POMDPs.create_action
import POMDPs.create_state
import POMDPs.create_transition
import POMDPs.create_observation
import POMDPs.transition!
import POMDPs.observation!
import POMDPs.reward
import POMDPs.n_states
import POMDPs.n_actions
import POMDPs.n_observations
import POMDPs.rand!

import POMDPs: create_interpolants, interpolants!, weight, index
import POMDPs: dimensions

#=
export 
    TigerPOMDP,
    TigerState,
    TigerAction,
    TigerObservation,
    TransitionDistribution,
    ObservationDistribution,
    create_state,
    create_action,
    create_transition,
    create_observation,
    n_states,
    n_actions,
    n_observations,
    states,
    actions!,
    transition!
    observation!,
    reward,
    rand!,
    interpolants!
=#

type TigerPOMDP <: POMDP
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
end

type TigerState
    tigerleft::Bool
end

type TigerObservation
    obsleft::Bool
end

# Incompatible until Julia 0.4: @enum TigerAction listen=1 openleft=2 openright=3

abstract Enum
immutable TigerAction <: Enum
    val::Int
    function TigerAction(i::Integer)
        @assert 1 <= i <= 3
        new(i)
    end
end

==(x::TigerAction, y::TigerAction) = x.val == y.val

const listen = TigerAction(1)
const openleft = TigerAction(2)
const openright = TigerAction(3)

type TransitionDistribution <: AbstractDistribution
    isleft::Bernoulli
    TransitionDistribution() = new(Bernoulli(0.5))
end


type ObservationDistribution <: AbstractDistribution
    growlsleft::Bernoulli
    ObservationDistribution() = new(Bernoulli(0.85))
end


create_state(::TigerPOMDP) = TigerState(false)
create_action(::TigerPOMDP) = TigerAction(1)
create_transition(::TigerPOMDP) = TransitionDistribution()
create_observation(::TigerPOMDP) = ObservationDistribution()

n_states(::TigerPOMDP) = 2
n_actions(::TigerPOMDP) = 3
n_observations(::TigerPOMDP) = 2


# Resets the problem after opening door; does nothing after listening
function transition!(d::TransitionDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    if a == openleft || a == openright
        d.isleft = Bernoulli(0.5)
    end
    d
end

function observation!(d::ObservationDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    if a == listen
        if s.tigerleft
            d.growlsleft = Bernoulli(0.85)
        else
            d.growlsleft = Bernoulli(0.15)
        end
    end
    d
end

function reward(pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    r = 0.0
    if a == listen
        r += pomdp.r_listen
    end
    if a == openleft
        if s.tigerleft
            r += pomdp.r_findtiger
        else
            r += pomdp.r_escapetiger
        end
    end
    if a == openright
        if s.tigerleft
            r += pomdp.r_escapetiger
        else
            r += pomdp.r_findtiger
        end
    end
    return r
end

function rand!(s::TigerState, d::TransitionDistribution)
    s.tigerleft = rand(d.isleft)
    s
end

function rand!(s::TigerState, d::ObservationDistribution)
    s.tigerleft = rand(d.obsleft)
    s
end

dimensions(::ObservationDistribution) = 1 ## each condition gives 1D, but there are two states...?
dimensions(::TransitionDistribution) = 1

function states(::TigerPOMDP)
    [TigerState(i) for i = 0:1]
end

function actions!(acts::Vector{TigerAction}, ::TigerPOMDP, s::TigerState)
    acts[1:end] = [listen, openleft, openright]
end

create_interpolants(::TigerPOMDP) = Interpolants()

function interpolants!(interpolants::Interpolants, d::TransitionDistribution)
    empty!(interpolants)
    ph = params(d.isleft)[1]
    push!(interpolants, 1, (1-ph)) # tiger on left
    push!(interpolants, 2, (ph)) # tiger on right
    interpolants
end

function interpolants!(interpolants::Interpolants, d::ObservationDistribution)
    empty!(interpolants)
    ph = params(d.growlsleft)[1]
    push!(interpolants, 1, (1-ph)) # hear growling from left
    push!(interpolants, 2, (ph)) # hear growling from right
    interpolants
end

length(interps::Interpolants) = interps.length

weight(interps::Interpolants, i::Int64) = interps.weights[i]

index(interps::Interpolants, i::Int64) = interps.indices[i]

function convert!(x::Vector{Float64}, state::TigerState)
    x[1] = float(state.tigerleft)
    x
end


#end #module
