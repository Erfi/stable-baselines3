class MBCTRL:
    def __init__(self, *args, **kwargs):
        """Creates class instance."""
        pass

    def train(self, states, next_states, actions):
        """Trains this controller."""
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller."""
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, state):
        """Performs an action."""
        raise NotImplementedError("Must be implemented in subclass.")
