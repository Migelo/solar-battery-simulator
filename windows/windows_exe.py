"""Entry point of the Windows .exe (built by windows/build_exe.py with PyInstaller).

Double-clicking the exe starts the GUI on 127.0.0.1 and opens the browser. data/, the
price cache and the simulation cache live next to the exe, whatever folder it is started from.
"""

import multiprocessing
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    print("Solar Battery Simulator is starting - your browser will open in a moment.")
    print("Keep this window open while you use it; close it to stop the simulator.\n")

    from solar_battery_simulator import gui

    gui.main()
