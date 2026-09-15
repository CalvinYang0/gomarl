#!/usr/bin/env python3
"""Submit the selected Advantage-mask profiles on memory-safe corridor."""
import os

os.environ.setdefault("TARGET_SCENE", "corridor")

from ozstar_submit_advantage_3s5z_safe import main


if __name__ == "__main__":
    main()
