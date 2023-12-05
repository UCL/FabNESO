"""Neptune Exploratory SOftware (NESO) plugin for FabSim3."""

try:
    from .tasks import neso, neso_ensemble

    __all__ = ["neso", "neso_ensemble"]

except ImportError:
    import warnings

    warnings.warn(
        "Cannot import FabNESO tasks - fabsim.base.fab may not be importable",
        stacklevel=2,
    )
