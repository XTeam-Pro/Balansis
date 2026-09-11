"""Build the optional kernel; a requested native build must fail visibly."""

import hashlib
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class StrictBuildExt(build_ext):
    def build_extensions(self):
        # Compiler flags and source identity can change without a C file mtime.
        self.force = True
        if self.compiler.compiler_type == "msvc":
            flags = ["/O2", "/fp:strict", "/std:c11"]
        else:
            flags = [
                "-O3",
                "-std=c11",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fexcess-precision=standard",
            ]
        for extension in self.extensions:
            extension.extra_compile_args = flags
            # A linker fast-math flag can install process-wide FTZ/DAZ startup
            # code, even when each compilation used strict arithmetic.
            extension.extra_link_args = (
                []
                if self.compiler.compiler_type == "msvc"
                else ["-fno-fast-math", "-ffp-contract=off"]
            )
        super().build_extensions()


source_hash = hashlib.sha256()
for name in [
    "pyproject.toml",
    "setup.py",
    "src/module.c",
    "src/exact_dot.c",
    "src/exact_dot.h",
    "src/neumaier.c",
    "src/neumaier.h",
]:
    source_hash.update(name.encode() + b"\0" + Path(name).read_bytes())

setup(
    ext_modules=[
        Extension(
            "_balansis_kernels",
            ["src/module.c", "src/neumaier.c", "src/exact_dot.c"],
            depends=["src/neumaier.h", "src/exact_dot.h"],
            define_macros=[
                ("BALANSIS_SOURCE_SHA256", '"' + source_hash.hexdigest() + '"')
            ],
        )
    ],
    cmdclass={"build_ext": StrictBuildExt},
)
