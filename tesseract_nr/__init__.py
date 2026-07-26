"""Numerical-relativity PDE backend for Tesseract.

This package contains the standalone CPU reference implementation used to
qualify the coupled CCZ4, relativistic matter, Carter two-current, and
mixed-vector evolution before individual kernels are promoted to Vulkan.
"""

from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .convergence import run_constitutive_convergence
from .constrained_convergence import run_constrained_convergence
from .frozen_closure import QualifiedFrozenClosure
from .grid import CartesianGrid, GhostZonePatch, PeriodicGrid
from .matter import FluidPrimitive, IdealGasEOS
from .production33 import (
    Theory33ProductionSolver,
    Theory33Recovery,
    Theory33RecoveryReport,
    load_theory33_production_state,
    save_theory33_production_state,
)
from .theory33 import (
    CarrierPrimitive,
    CharacteristicAudit,
    MasterState,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


__all__ = [
    "CCZ4Parameters",
    "CCZ4Solver",
    "CCZ4State",
    "QualifiedFrozenClosure",
    "run_constitutive_convergence",
    "run_constrained_convergence",
    "CartesianGrid",
    "GhostZonePatch",
    "PeriodicGrid",
    "FluidPrimitive",
    "IdealGasEOS",
    "Theory33ProductionSolver",
    "Theory33Recovery",
    "Theory33RecoveryReport",
    "load_theory33_production_state",
    "save_theory33_production_state",
    "CarrierPrimitive",
    "CharacteristicAudit",
    "MasterState",
    "Theory33MasterFunction",
    "Theory33MasterParameters",
]

__version__ = "0.1.0"
