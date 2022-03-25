

from abc import ABC, abstractmethod


class Metric (object):

    def __init__(self) -> None:
        self.__name = 'metric'
    
    def __call__(self, src, dsc):
        return self._process(src, dsc)

    @abstractmethod
    def _process(src, dsc):
        pass

    def __str__(self) -> str:
        return self._name
     