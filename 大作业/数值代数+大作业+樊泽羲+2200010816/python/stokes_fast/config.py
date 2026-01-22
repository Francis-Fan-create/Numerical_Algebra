"""JAX configuration (double precision)."""

from jax import config as _cfg

_cfg.update("jax_enable_x64", True)
