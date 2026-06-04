import sys
import importlib
import random
import numpy as np
import torch

def dynamic_module_reloader(
        modules: str|list[str] = "",
        verbose: bool = False
        ) -> None:
    """
    Reload all relevant modules dynamically.
    
    Args:
        modules: The modules to reload. Can be a single module or a list of modules.
        verbose: Whether to print the names of the reloaded modules.
    """
    if isinstance(modules, str):
        modules = [modules]
    for name in modules:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
                if verbose:
                    print(f"Reloaded: {name}")
            except Exception as e:
                print(f"Warning: Could not reload {name}: {e}")


def seed_everything(seed: int) -> None:
    """
    Seeds the random number generators.

    Args:
        seed: The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")