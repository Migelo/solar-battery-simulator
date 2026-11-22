"""Build the single-file Windows app: ``uv run --with pyinstaller python windows/build_exe.py``.

Writes ``dist/SolarBatterySimulator.exe`` (on macOS/Linux a native binary of the same name,
useful only for local checks: PyInstaller cannot cross-compile).
"""

import os
from pathlib import Path

import nicegui
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "src" / "solar_battery_simulator"

PyInstaller.__main__.run(
    [
        str(HERE / "windows_exe.py"),
        "--name=SolarBatterySimulator",
        "--onefile",
        "--noconfirm",
        "--clean",
        # NiceGUI serves its static JS/CSS from the package directory.
        f"--add-data={Path(nicegui.__file__).parent}{os.pathsep}nicegui",
        # Default network tariff table (read next to gui.py) when none sits next to the exe.
        f"--add-data={PACKAGE / 'cenikOmreznin.csv'}{os.pathsep}solar_battery_simulator",
        f"--paths={PACKAGE.parent}",
        # pandas imports the Excel readers lazily, so PyInstaller can't see them.
        "--hidden-import=openpyxl",
        "--hidden-import=xlrd",
        "--hidden-import=solar_battery_simulator.gui",
    ]
)
