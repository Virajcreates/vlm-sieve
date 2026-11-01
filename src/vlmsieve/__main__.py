"""Enable python -m vlmsieve execution."""
from __future__ import annotations

import sys

from vlmsieve.cli import main

if __name__ == "__main__":
    sys.exit(main())
