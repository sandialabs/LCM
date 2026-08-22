"""Site factory registration for driving LCM/Albany from MatCal.

Imported for its side effects by ``site_matcal/__init__.py``. For a plain
external-executable model producing Exodus output, MatCal's defaults already
work (the Exodus importer defaults to exodus_helper, which is installed), so
this module only makes the environment deterministic:

  * clears the executable-environment-setup registry so the model runs in a
    clean shell with no injected `module load` / site commands (matters on a
    machine where another site_matcal registered a Sierra environment), and
  * forces the plain jinja2 template processor (rather than pyprepro, which a
    Sandia site_matcal may register).

Both operations are idempotent and safe to run at every import.
"""

from matcal.core.file_modifications import use_jinja_preprocessor
from matcal.core.external_executable import (
    matcal_executable_environment_setup_function_identifier,
)


def register():
    # Clean shell environment: fall back to default_environment_command_processor.
    matcal_executable_environment_setup_function_identifier._registry = {}
    # Plain jinja2 templating of input decks.
    use_jinja_preprocessor()


register()
