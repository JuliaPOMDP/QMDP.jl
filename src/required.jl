const REQUIRED_FUNCTIONS = [states,
                            actions,
                            iterator,
                            create_transition_distribution,
                            transition,
                            reward,
                            pdf,
                            state_index]

function required()
    POMDPs.get_methods(REQUIRED_FUNCTIONS)
end
