"""Platform-wheel hooks; project metadata lives in pyproject.toml."""

from pathlib import Path
import runpy

from setuptools import setup


_hooks = runpy.run_path(str(Path(__file__).resolve().parent / "build_rust.py"))


setup(
    cmdclass={
        "bdist_wheel": _hooks["PlatformWheel"],
        "build_py": _hooks["RustBuildPy"],
    },
    distclass=_hooks["BinaryDistribution"],
)
