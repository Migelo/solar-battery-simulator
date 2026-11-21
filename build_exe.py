"""Build the single-file Windows app: ``uv run --with pyinstaller python build_exe.py``.

Writes ``dist/SolarBatterySimulator.exe`` (on macOS/Linux a native binary of the same name,
useful only for local checks: PyInstaller cannot cross-compile).
"""

import os
from pathlib import Path

import nicegui
import PyInstaller.__main__

PyInstaller.__main__.run(
    [
        "windows_exe.py",
        "--name=SolarBatterySimulator",
        "--onefile",
        "--noconfirm",
        "--clean",
        # NiceGUI serves its static JS/CSS from the package directory.
        f"--add-data={Path(nicegui.__file__).parent}{os.pathsep}nicegui",
        # pandas imports the Excel readers lazily, so PyInstaller can't see them.
        "--hidden-import=openpyxl",
        "--hidden-import=xlrd",
        "--hidden-import=gui",
    ]
)
