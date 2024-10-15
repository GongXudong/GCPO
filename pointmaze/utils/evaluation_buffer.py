from typing import List


class EvaluationBuffer(object):

    def __init__(self, max_capacity: int=10000, decay_coef: float=0.99) -> None:
        self.max_capacity = max_capacity
        self.decay_coef = decay_coef

        self._evaluation_stat: List = []

    def merge(self, new_evaluation_info: List):
        self._evaluation_stat.extend(new_evaluation_info)
        if len(self._evaluation_stat) > self.max_capacity:
            self._evaluation_stat = self._evaluation_stat[-self.max_capacity:]
        