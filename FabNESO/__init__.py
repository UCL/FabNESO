"""Neptune Exploratory SOftware (NESO) plugin for FabSim3."""

try:
    from .tasks import (
        neso,
        neso_grid_ensemble,
        neso_pce_analysis,
        neso_pce_ensemble,
        neso_qmc_ensemble,
        neso_vbmc,
        neso_write_field,
    )

    __all__ = [
        "neso",
        "neso_grid_ensemble",
        "neso_pce_ensemble",
        "neso_pce_analysis",
        "neso_qmc_ensemble",
        "neso_vbmc",
        "neso_write_field",
    ]

except ImportError:
    import warnings

    warnings.warn(
        "Cannot import FabNESO tasks - fabsim.base.fab may not be importable",
        stacklevel=2,
    )
