"""Run the closed loop: ``python -m closed_loop --help``."""

import sys

from closed_loop.cli import main

if __name__ == "__main__":
    sys.exit(main())
