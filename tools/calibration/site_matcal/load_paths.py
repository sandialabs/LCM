"""Registry of cap-model calibration load paths and comparison curves.

A **load path** is one single-element Albany deck. All paths drive the *same*
cap parameter set (the whole point of multi-path calibration): the parameters
are physical material constants, so one set must fit hydrostatic, confined and
triaxial responses simultaneously.

  * hydrostatic - equal compression on all three axes; cap-only response
                  (constrains R, W, D1, kappa0, calpha).
  * confined    - 1D/oedometric compression, axial x with the lateral strains
                  held at zero; shear + cap interaction.
  * triaxial    - three unequal compressive strains, so psi<1 and the
                  non-associative terms are active (constrains A, C, theta,
                  psi, L, phi, Q). Uses the CapModelTriaxial materials file
                  (same placeholders, different block/material name).

Each path records its ``axis``: the direction the deck loads hardest, which is
the axial direction an experiment would report. It is ``x`` for all three
decks (hydrostatic loads all three axes equally, so any axis would do).

A **curve** says which pair of fields to compare, and is what makes the
harness usable with whatever the laboratory measured:

  * ``stress-strain``    - axial strain vs axial Cauchy stress (Pa). The
                           default, and the usual output of a material test.
  * ``load-displacement`` - face displacement (m) vs face reaction force (N).
  * ``time-stress``      - LOCA continuation parameter vs axial stress. `time`
                           runs over [0,1] and is affine in applied strain, so
                           this is equivalent to ``stress-strain`` up to a
                           rescaling of the abscissa. Kept for regression
                           checks that want the abscissa the deck stepped on
                           rather than a measured one.

Both fields of a curve are produced by ``site_matcal.exodus_reader`` and must
appear, under these names, as columns of any experimental CSV.
"""


class Curve:
    """A pair of fields to compare, parameterized by the load path's axis."""

    def __init__(self, name, independent, dependent, units, description):
        self.name = name
        self._independent = independent
        self._dependent = dependent
        self.units = units
        self.description = description

    def fields(self, axis):
        """Return ``(independent_field, dependent_field)`` for ``axis``."""
        return (self._independent.format(a=axis), self._dependent.format(a=axis))


CURVES = {
    "stress-strain": Curve(
        "stress-strain", "strain_{a}{a}", "stress_{a}{a}",
        units=("dimensionless", "Pa"),
        description="axial strain vs axial Cauchy stress"),
    "load-displacement": Curve(
        "load-displacement", "displacement_{a}", "force_{a}",
        units=("m", "N"),
        description="loaded-face displacement vs reaction force"),
    "time-stress": Curve(
        "time-stress", "time", "stress_{a}{a}",
        units=("dimensionless", "Pa"),
        description="LOCA continuation parameter vs axial stress"),
}

DEFAULT_CURVE = "stress-strain"


class LoadPath:
    def __init__(self, name, deck, materials, exodus, axis):
        self.name = name
        self.deck = deck
        self.materials = materials
        self.exodus = exodus
        self.axis = axis

    def fields(self, curve=DEFAULT_CURVE):
        """Return ``(independent_field, dependent_field)`` for ``curve`` on
        this path's axial direction."""
        return get_curve(curve).fields(self.axis)


LOAD_PATHS = {
    "hydrostatic": LoadPath(
        "hydrostatic", "input_hydrostatic.yaml", "materials.yaml",
        "cap_hydrostatic.exo", axis="x"),
    "confined": LoadPath(
        "confined", "input_confined.yaml", "materials.yaml",
        "cap_confined.exo", axis="x"),
    "triaxial": LoadPath(
        "triaxial", "input_triaxial.yaml", "materials_triaxial.yaml",
        "cap_triaxial.exo", axis="x"),
}


def get_load_path(name):
    try:
        return LOAD_PATHS[name]
    except KeyError:
        raise KeyError(f"unknown load path {name!r}; known: {sorted(LOAD_PATHS)}")


def get_curve(name):
    try:
        return CURVES[name]
    except KeyError:
        raise KeyError(f"unknown curve {name!r}; known: {sorted(CURVES)}")
