import torch

def conditional_grad(dec):
    def decorator(func):
        if torch.is_grad_enabled():
            return dec(func)
        return func
    return decorator
