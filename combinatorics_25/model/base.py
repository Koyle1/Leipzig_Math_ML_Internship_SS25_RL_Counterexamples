class ModelType():
    registry ={}
    
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