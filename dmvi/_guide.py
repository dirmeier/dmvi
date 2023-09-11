import abc

import chex


class Guide(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, sample_shape=(1,)) -> chex.Array:
        pass

    @abc.abstractmethod
    def evidence(self, z) -> chex.Array:
        pass
