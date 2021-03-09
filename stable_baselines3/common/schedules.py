import math


class Schedule:
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class CosineSchedule(Schedule):
    def __init__(self, initial_value: float, scale: float = 1.0):
        """
        :param initial_value: starting value for the schedule
        :param scale: scale value for step.
            e.g. scale=2.0 reaches the end value of the schedule in twice fewer steps
            e.g. scale=0.5 reaches the end value of the schedule 2 times more steps (hence never reaching it)
        """
        self.initial_value = initial_value
        self.scale = scale

    def value(self, step: int, total_steps: int):
        """
        Step will be capped at total_steps
        """
        value = self.initial_value * (
            0.5 + (0.5 * math.cos(min(int(self.scale * step), total_steps) / total_steps * math.pi))
        )
        return value


class ConstantOneStep(Schedule):
    """
    One step schedule. Initial value upto total_steps and then zero
    """

    def __init__(self, initial_value: float, scale: float = 1.0):
        """
        :param initial_value: starting value for the schedule
        :param scale: scale value for step.
            e.g. scale=2.0 reaches the end value of the schedule in twice fewer steps
            e.g. scale=0.5 reaches the end value of the schedule 2 times more steps (hence never reaching it)
        """
        self.initial_value = initial_value
        self.scale = scale

    def value(self, step: int, total_steps: int):
        if int(self.scale * step) <= total_steps:
            return self.initial_value
        else:
            return 0.0
