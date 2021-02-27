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
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def value(self, step: int, total_steps: int):
        value = self.initial_value * (0.5 + (0.5 * math.cos(step / total_steps * math.pi)))
        return value
