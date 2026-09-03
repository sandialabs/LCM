"""Per-curve residual weighting for a multi-field comparison.

A curve that compares several dependent fields at once (``--curve
dev-stress-volumetric``) becomes one objective PER FIELD, collected into a
single evaluation set so the model still runs once per parameter set. Each
objective conditions its own field onto its own data range, so a stress in Pa
and a dimensionless strain already contribute comparably; this module is for
deliberately tilting that balance.

The reason it exists: volumetric strain is the least reliable quantity a
frozen-soil triaxial test produces. Xu's thesis cites a constitutive-model
workshop in which no model out of thirty two scored above 40 out of 100 at
predicting volumetric deformation. Weighted equally with the deviatoric
response, it would be free to pull the failure-envelope parameters, which the
deviatoric curve does constrain well, toward fitting its own noise.

One objective per field rather than one objective over several fields is not a
stylistic choice. MatCal's per-field weighting hook, ``UserFunctionWeighting``,
takes a plain function and re-imports it from its SOURCE
(``PythonLocalFunctionImporter``), so a closure carrying the scale factor
arrives with the factor undefined: it fails at evaluation time with
``NameError: name 'factor' is not defined``. ``ConstantFactorWeighting`` scales
a whole objective and is an object, so it survives.
"""

from matcal.core.residuals import ConstantFactorWeighting


def make_field_weight(factor):
    """Return a weighting that scales one field-objective's residual."""
    if factor <= 0.0:
        raise ValueError(f"a field weight must be positive, got {factor}")
    return ConstantFactorWeighting(factor)
