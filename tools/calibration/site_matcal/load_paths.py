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
  * txc         - conventional triaxial compression: consolidate to a
                  confining pressure, then shear axially with that pressure
                  held. This is the laboratory test, so it is the path that
                  takes measured triaxial data. Unlike the other three it
                  prescribes only the axial direction; the lateral faces carry
                  tractions and their displacement is predicted, which is what
                  lets the path see dilatancy (L, phi, Q) as well as the shear
                  envelope (A, C, D, theta).

Each path records its ``axis``: the direction the deck loads hardest, which is
the axial direction an experiment would report. It is ``x`` for all four decks
(hydrostatic loads all three axes equally, so any axis would do).

A path may also carry ``constants``: deck placeholders that are not material
parameters but describe the test, such as the confining pressure. They are
rendered into the deck like any other jinja value and can be overridden with
``--set``, so one deck covers a whole series of confining pressures. Only
``txc`` has any; for the other three the dict is empty and nothing changes.

``preload_time`` is the continuation parameter at which loading proper begins.
It is nonzero only for ``txc``, whose run starts with a consolidation stage;
the Exodus reader trims that stage off and refers the strains to the state at
the end of it. See ``site_matcal.exodus_reader``.

A **curve** says which pair of fields to compare, and is what makes the
harness usable with whatever the laboratory measured. Under finite deformation
the strain and stress measures stop being interchangeable, so the curve names
say which measure they mean:

  * ``true-stress-strain`` - logarithmic (true) strain vs Cauchy (true) stress
                           (Pa). The default: this is the pair the
                           finite-deformation kernel actually works in, since
                           it integrates in logarithmic elastic strain and
                           reports Cauchy stress.
  * ``eng-stress-strain``  - engineering strain ``u/L0`` vs engineering stress
                           ``force/A0``, both referred to the undeformed
                           geometry. This is what most laboratory reports
                           contain.
  * ``load-displacement``  - face displacement (m) vs face reaction force (N).
                           The raw measured quantities, identical in meaning
                           under both kinematics, so the safest choice when
                           the reduction to stress and strain is in doubt.
  * ``dev-stress-strain``  - engineering strain ``u/L0`` vs the deviatoric
                           (differential) Cauchy stress
                           ``sigma_aa - (sigma_bb + sigma_cc)/2`` (Pa). On a
                           triaxial path with equal lateral stresses that is
                           exactly the ``q = sigma_1 - sigma_3`` a triaxial
                           laboratory reports, so this is the curve to use
                           with ``txc`` data. It needs no confining pressure
                           in the data file, since the confinement cancels out
                           of the difference.
  * ``time-stress``        - LOCA continuation parameter vs axial Cauchy
                           stress. `time` runs over [0,1] and is affine in
                           applied displacement. Kept for regression checks
                           that want the abscissa the deck stepped on rather
                           than a measured one.

Under ``Finite Deformation: false`` the two stress-strain curves differ only by
the difference between ``u/L0`` and ``ln(1 + u/L0)``, because the face area
does not change. Under finite deformation they differ in the stress as well,
on any path whose loaded face changes area.

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
    "true-stress-strain": Curve(
        "true-stress-strain", "strain_log_{a}", "stress_{a}{a}",
        units=("dimensionless", "Pa"),
        description="logarithmic (true) strain vs Cauchy (true) stress"),
    "eng-stress-strain": Curve(
        "eng-stress-strain", "strain_eng_{a}", "stress_eng_{a}",
        units=("dimensionless", "Pa"),
        description="engineering strain u/L0 vs engineering stress force/A0"),
    "load-displacement": Curve(
        "load-displacement", "displacement_{a}", "force_{a}",
        units=("m", "N"),
        description="loaded-face displacement vs reaction force"),
    "dev-stress-strain": Curve(
        "dev-stress-strain", "strain_eng_{a}", "stress_dev_{a}",
        units=("dimensionless", "Pa"),
        description="engineering strain u/L0 vs deviatoric (differential) "
                    "Cauchy stress sigma_1 - sigma_3"),
    "time-stress": Curve(
        "time-stress", "time", "stress_{a}{a}",
        units=("dimensionless", "Pa"),
        description="LOCA continuation parameter vs axial Cauchy stress"),
}

DEFAULT_CURVE = "true-stress-strain"


class LoadPath:
    def __init__(self, name, deck, materials, exodus, axis, constants=None,
                 preload_constant=None):
        self.name = name
        self.deck = deck
        self.materials = materials
        self.exodus = exodus
        self.axis = axis
        # Deck placeholders describing the test rather than the material.
        self.constants = dict(constants or {})
        # Which of them, if any, holds the continuation parameter at the end of
        # the preload stage. Named rather than stored directly so that a --set
        # override of it reaches the Exodus reader too.
        self.preload_constant = preload_constant

    def fields(self, curve=DEFAULT_CURVE):
        """Return ``(independent_field, dependent_field)`` for ``curve`` on
        this path's axial direction."""
        return get_curve(curve).fields(self.axis)

    def preload_time(self, constants=None):
        """The continuation parameter at which loading proper begins, given the
        constants actually in force (defaults merged with any ``--set``
        overrides). Zero for every path that loads from the undeformed state."""
        if self.preload_constant is None:
            return 0.0
        merged = {**self.constants, **(constants or {})}
        return float(merged[self.preload_constant])


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
    "txc": LoadPath(
        "txc", "input_txc.yaml", "materials.yaml",
        "cap_txc.exo", axis="x",
        constants=dict(
            # Cell pressure, Pa, as a positive magnitude. Held constant through
            # the shear stage. One --set per curve of a confining-pressure
            # series.
            confining_pressure=1.0e6,
            # Axial engineering strain applied during the shear stage,
            # negative in compression, referred to the consolidated length.
            # Cover the strain range of the data: a run that stops short of it
            # gives MatCal nothing to interpolate onto at the far end.
            axial_strain=-0.20,
            # Fraction of the continuation run spent consolidating. Only large
            # enough to resolve the ramp; the reader trims it off.
            preload_fraction=0.1,
        ),
        preload_constant="preload_fraction"),
}

# Every deck placeholder that is not a material parameter, across all paths,
# with its default. The command line uses this to decide whether a --set name
# is legitimate and to report the defaults in --help.
DECK_CONSTANTS = {name: value
                  for path in LOAD_PATHS.values()
                  for name, value in path.constants.items()}


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
