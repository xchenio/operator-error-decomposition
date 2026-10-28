"""Source package.

`__all__` lists submodules, not exported symbols. The last group is the current
public-PDEBench pipeline; `config`, `trainer`, `basis`, `diagnostics`, `data` and `pdes`
belong to the withdrawn synthetic basis-align study and are retained for provenance only.
"""

__all__ = [
    # shared, used by both pipelines
    "utils", "derivatives", "models", "resume",
    # current pipeline: public PDEBench mean/fluctuation study
    "public_benchmark", "fno", "fno1d_multi", "fno2d", "fno2d_multi",
    "projection_audit", "health_checks", "source_hash",
    # withdrawn basis-align pipeline, kept for provenance
    "config", "trainer", "basis", "diagnostics", "data", "pdes",
]
