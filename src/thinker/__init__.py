"""thinker: reasoning model evaluation and training toolkit.

Importing this package loads a .env found by walking up from the current
working directory, so credentials are available to any submodule without an
explicit setup call. Guarded so ``pip install --no-deps`` installs still import.
"""

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from .config import get_configs

__all__ = ["get_configs"]
