import os

def n_params(model):
    """ Calculate total number of parameters in a model.
    Args:
        model: nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def safe_create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)