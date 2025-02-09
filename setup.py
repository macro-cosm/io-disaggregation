from setuptools import find_packages, setup

setup(
    name="multi_sector_disagg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "cvxpy>=1.2.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Jose Moran",
    description="A package for multi-sector input-output table disaggregation",
    license="MIT",
)
