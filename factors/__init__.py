# factors/__init__.py
"""
Factor Registry — auto-discovers all factor modules.
Each factor module must expose a `signal(df, n, factor_name)` function.
"""
import importlib
import pkgutil
from pathlib import Path

FACTOR_REGISTRY = {}

# Auto-discover all .py files in this directory
_package_dir = Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([str(_package_dir)]):
    if module_name.startswith('_'):
        continue
    module = importlib.import_module(f'.{module_name}', package=__name__)
    if hasattr(module, 'signal'):
        FACTOR_REGISTRY[module_name] = module.signal

def compute_factor(df, factor_name: str, n: int, output_col: str = None):
    """
    Universal factor computation interface.
    Usage: compute_factor(df, 'bias_q', 60, 'bias_q_60')
    """
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor '{factor_name}'. Available: {list(FACTOR_REGISTRY.keys())}")
    
    if output_col is None:
        output_col = f"{factor_name}_{n}"
    
    return FACTOR_REGISTRY[factor_name](df, n, output_col)