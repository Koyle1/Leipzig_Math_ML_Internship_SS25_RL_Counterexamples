from abc import ABC, abstractmethod
import numpy as np

class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
      
  @abstractmethod
  def __repr__(self):
    pass

  @classmethod
  def register(cls, name):
      def inner(subclass):
          cls.registry[name.lower()] = subclass
          return subclass
      return inner

  @classmethod
  def create(cls, name, *args, **kwargs):
      subclass = cls.registry.get(name.lower())
      if subclass is None:
          raise ValueError(f"Unknown model name: {name}")
      return subclass(*args, **kwargs)

@BaseFunction.register("BaseConjuncture1")
class BaseConjuncture1(BaseFunction):

    def __init__():
        pass

    def get(self, env, dist, info):
        pass

    def __repr__(self):
        return "BaseConjuncture1"

@BaseFunction.register("BaseConjuncture2")
class BaseConjuncture2(BaseFunction):

    def __init__():
        pass

    def get(self, env, dist, info):
        pass

    def __repr__(self):
        return "BaseConjuncture2"