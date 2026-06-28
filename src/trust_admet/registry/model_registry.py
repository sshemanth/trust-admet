MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model: {name}. Available models: {available}")
    return MODEL_REGISTRY[name]


def list_models():
    return sorted(MODEL_REGISTRY.keys())
