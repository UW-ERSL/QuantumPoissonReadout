from setuptools import setup

setup(
    name="quantumpoissonreadout",
    version="0.1.0",
    description="Least-squares quantum readout of the effective conductivity of periodic two-phase cells",
    url="https://github.com/UW-ERSL/QuantumPoissonReadout",
    packages=["quantumpoissonreadout"],
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "qiskit>=1.0", "pyqsp",
                      "quantuminversionpoisson @ git+https://github.com/UW-ERSL/QuantumInversionPoisson"],
)
