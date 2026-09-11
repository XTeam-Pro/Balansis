"""Build the TNSIM distribution from its own package root."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="tnsim",
    version="1.1.0",
    python_requires=">=3.10",
    author="Andrey Tikhonov",
    author_email="andrew@xteam.pro",
    description="Finite series computation and a FastAPI service",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="AGPL-3.0-only",
    license_files=[
        "LICENSE",
        "NOTICE",
        "LICENSING.md",
        "COMMERCIAL_LICENSE.md",
        "ORDER_FORM_TEMPLATE.md",
        "SECURITY.md",
    ],
    url="https://github.com/StudyLabPro/Balansis",
    packages=["tnsim"]
    + [
        "tnsim." + name
        for name in find_packages(str(ROOT), exclude=["tests", "tests.*"])
    ],
    package_dir={"tnsim": "."},
    package_data={"tnsim": ["migrations/*.sql", "database/init.sql"]},
    install_requires=["numpy>=1.24,<2", "pydantic>=2.5,<3"],
    extras_require={
        "api": ["fastapi>=0.104,<1", "uvicorn>=0.24,<1", "asyncpg>=0.29,<1"],
        "torch": ["torch>=2"],
        "dev": ["pytest>=7", "pytest-asyncio>=0.21", "httpx>=0.25"],
    },
    entry_points={"console_scripts": ["tnsim-server=tnsim:run_server"]},
)
