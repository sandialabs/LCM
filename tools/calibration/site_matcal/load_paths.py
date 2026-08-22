"""Registry of cap-model calibration load paths.

Each load path is one single-element Albany deck plus the fields to compare.
All paths drive the *same* cap parameter set (the whole point of multi-path
calibration): the parameters are physical material constants, so one set must
fit hydrostatic, confined and triaxial responses simultaneously.

  * hydrostatic - cap-only response (constrains R, W, D1, kappa0, calpha).
  * confined    - 1D/oedometric compression; shear + cap interaction.
  * triaxial    - psi<1 + non-associative terms active (constrains A, C,
                  theta, psi, L, phi, Q). Uses the CapModelTriaxial materials
                  file (same placeholders, different block/material name).

`independent`/`dependent` name fields the exodus reader returns; `time` in
[0,1] is the LOCA continuation parameter and is affine in applied strain, so a
curve objective on `time` is equivalent to one on strain for these decks.
"""


class LoadPath:
    def __init__(self, name, deck, materials, exodus, independent, dependent):
        self.name = name
        self.deck = deck
        self.materials = materials
        self.exodus = exodus
        self.independent = independent
        self.dependent = dependent


LOAD_PATHS = {
    "hydrostatic": LoadPath(
        "hydrostatic", "input_hydrostatic.yaml", "materials.yaml",
        "cap_hydrostatic.exo", independent="time", dependent="stress_zz"),
    "confined": LoadPath(
        "confined", "input_confined.yaml", "materials.yaml",
        "cap_confined.exo", independent="time", dependent="stress_zz"),
    "triaxial": LoadPath(
        "triaxial", "input_triaxial.yaml", "materials_triaxial.yaml",
        "cap_triaxial.exo", independent="time", dependent="stress_zz"),
}


def get_load_path(name):
    try:
        return LOAD_PATHS[name]
    except KeyError:
        raise KeyError(f"unknown load path {name!r}; known: {sorted(LOAD_PATHS)}")
