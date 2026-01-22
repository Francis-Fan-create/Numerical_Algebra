"""Accelerated Stokes solvers (JAX backend, reorganized)."""

from . import config as _config
from .fields import rhs_terms, velocity_error
from .solvers import solve_vcycle, solve_uzawa, solve_inexact_uzawa

__all__ = ["rhs_terms", "velocity_error", "solve_vcycle", "solve_uzawa", "solve_inexact_uzawa"]
