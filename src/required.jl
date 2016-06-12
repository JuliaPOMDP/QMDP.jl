const REQUIRED_FUNCTIONS = [states,
                            actions,
                            iterator,
                            create_transition_distribution,
                            transition,
                            reward,
                            pdf,
                            discount,
                            state_index]

function required_methods()
    POMDPs.get_methods(REQUIRED_FUNCTIONS)
end
