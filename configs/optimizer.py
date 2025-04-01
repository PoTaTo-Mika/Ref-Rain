import torch
from torch.optim import SGD, Adam, RMSprop, Adagrad, AdamW

def get_optimizer_name(name):
    """Get the optimizer class by name.
    
    Args:
        name (str): Name of the optimizer (case-insensitive).
        
    Returns:
        torch.optim.Optimizer: The optimizer class.
        
    Raises:
        ValueError: If the optimizer name is not supported.
    """
    optimizer_dict = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adamw': AdamW,
        #'muon' : Muon
    }
    
    name_lower = name.lower()
    if name_lower not in optimizer_dict:
        raise ValueError(f"Optimizer '{name}' not supported. Available options: {list(optimizer_dict.keys())}")
    
    return optimizer_dict[name_lower]