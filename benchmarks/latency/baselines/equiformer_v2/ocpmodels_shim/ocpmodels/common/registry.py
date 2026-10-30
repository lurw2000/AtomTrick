class _Registry:
    _models = {}

    def register_model(self, name):
        def decorator(cls):
            self._models[name] = cls
            return cls
        return decorator

    def get_model_class(self, name):
        return self._models[name]

registry = _Registry()
