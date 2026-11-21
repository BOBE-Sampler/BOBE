"""
Utility functions for managing global random seeds across the BOBE package.
"""

import os
import random
import numpy as np
import jax
import jax.random as jax_random
from .log import get_logger
log = get_logger("seed_utils")

# Global variables
_global_seed: int | None = None
_jax_key: jax.Array | None = None
_np_rng: np.random.Generator | None = None


def _ensure_seed_is_set():
    """Internal helper to initialize the global seed if it hasn't been set."""
    if _global_seed is None:
        log.info("Global seed not set. Initializing with a random seed.")
        set_global_seed()


def set_global_seed(seed: int | None = None) -> int:
    """Set global random seed for reproducible results.

    Args:
        seed: The random seed to use. If None, a random seed is generated.

    Returns:
        The seed that was used.
    """
    global _global_seed, _jax_key, _np_rng

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        log.info(f"No seed provided. Generated a random seed: {seed}")
    elif not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer or None")

    _global_seed = seed
    random.seed(seed)
    _np_rng = np.random.default_rng(seed)
    _jax_key = jax_random.PRNGKey(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    log.info(f"Global random seed set to {seed}")
    return seed


def get_global_seed() -> int:
    """Get the current global seed value.

    If the seed has not been set, it will be initialized automatically.
    
    Returns:
        The current global seed.
    """
    _ensure_seed_is_set()
    return _global_seed


def get_jax_key() -> jax.Array:
    """Get the current JAX random key.

    If the seed has not been set, it will be initialized automatically.

    Returns:
        The current JAX PRNGKey.
    """
    _ensure_seed_is_set()
    return _jax_key


def split_jax_key() -> tuple[jax.Array, jax.Array]:
    """Split the current JAX random key and update the global key.

    If the seed has not been set, it will be initialized automatically.

    Returns:
        A tuple containing the new global key and the key for use.
    """
    global _jax_key
    _ensure_seed_is_set()
    _jax_key, use_key = jax_random.split(_jax_key)
    return _jax_key, use_key


def get_new_jax_key() -> jax.Array:
    """Get a new JAX random key by splitting the current global key.

    If the seed has not been set, it will be initialized automatically.

    Returns:
        A new JAX PRNGKey for immediate use.
    """
    _, use_key = split_jax_key()
    return use_key


def get_numpy_rng() -> np.random.Generator:
    """Get the global NumPy random number generator.

    If the seed has not been set, it will be initialized automatically.

    Returns:
        The global instance of numpy.random.Generator.
    """
    _ensure_seed_is_set()
    return _np_rng

def ensure_reproducibility(seed: int | None = None) -> int:
    """Ensure reproducibility by setting seeds and JAX configurations.

    Args:
        seed: The seed to use. If None, a random seed will be generated.

    Returns:
        The seed that was used.
    """
    used_seed = set_global_seed(seed)
    log.info(f"Reproducibility ensured with seed {used_seed}")
    return used_seed