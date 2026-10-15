"""Entry point for PyInstaller and `python prototypes/gui_web/main.py` (same flags as
`python -m gui_web`)."""

import sys

from gui_web.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
