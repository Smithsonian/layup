<h1 align="center">
<img src="https://raw.githubusercontent.com/Smithsonian/layup/main/docs/images/layup_logo.png" width="500">
</h1><br>

# layup
Orbit fitting at LSST scale

[![ci](https://github.com/Smithsonian/layup/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/Smithsonian/layup/actions/workflows/smoke-test.yml)
[![pytest](https://github.com/Smithsonian/layup/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/Smithsonian/layup/actions/workflows/testing-and-coverage.yml)
[![Documentation Status](https://readthedocs.org/projects/layup/badge/?version=latest)](https://layup.readthedocs.io/en/latest/?badge=latest)
<!-- PyPI badge removed 2026-08-24. The `layup` name on PyPI currently holds only
     version 0.0.1, uploaded 2025-01-09 (four days before this repository's first
     commit): a 1.3 kB wheel containing an empty __init__.py, with no dependencies.
     `pip install layup` therefore succeeds and installs nothing, with no error.
     The badge advertised that as if it were the package.
     Restore this line once a real release is published -- see issue #436, which
     also covers the direct-URL sorcha dependency that currently prevents one.
[![PyPI - Version](https://img.shields.io/pypi/v/layup)](https://pypi.python.org/pypi/layup)
-->
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

## Setup

### Requirements

| | |
|---|---|
| Python | **3.11 or newer** |
| Compiler | a **C++17** compiler — `layup` builds a C++ extension. Xcode command line tools on macOS; `build-essential` or equivalent on Linux |
| pip | **21.3 or newer** (editable installs of `pyproject.toml`-only projects need PEP 660 support) |
| Platforms | macOS and Linux. Windows is not supported and is not tested |
| Disk | about **3.2 GB** free — roughly 1.5 GB of that is the ephemeris and reference data fetched by `layup bootstrap` |

If `pip install -e .` fails with *"File `setup.py` or `setup.cfg` not found"*, your
pip predates PEP 660: run `pip install --upgrade pip` first. Note that the
`python3` shipped with macOS is too old; install a newer Python before creating
the environment.

Before installing layup, it's a great idea to create a virtual environment with either `conda` or `venv`.

You can download the source code with:
```
git clone --recursive https://github.com/Smithsonian/layup.git
```

The `--recursive` flag matters: `layup` vendors **`eigen`** and **`autodiff`** as
git submodules under `include/`, and the C++ build fails with a bare
`fatal error: 'Eigen/Dense' file not found` if they are absent. If you already
cloned without it, run
```
git submodule update --init
```
(`assist` and `rebound` are *not* submodules — they are installed as ordinary
Python dependencies.)

Next, enter the layup directory and run
```
pip install -e .
```
to create an editable install of `layup`. If you're doing development work, you can install with
```
pip install -e ".[dev]"
```
to install all of the development packages as well.

### Running the tests
Run `layup bootstrap` first — a large fraction of the suite is skipped without the
ephemeris and reference data, so a run that has not bootstrapped will report
success while validating none of the orbit fitting. Then:
```
pytest
```

### Adding new submodule 
Note that to get the new submodules added in an existing copy of the repo you want to run
```
git submodule update --init
```
And in subsequent clones of the repo you want to run
```
git clone --recursive https://github.com/Smithsonian/layup.git
```

## Quickstart

Once `layup` is installed, download the ephemeris and reference data it needs
(SPICE planetary kernels, the small-body kernel, MPC observatory codes, and the
astrometry debiasing tables). This is a one-time download of roughly 1 GB, which
expands to about 1.5 GB on disk:
```
layup bootstrap
```

### Fit an orbit from the command line

`layup` bundles a demo dataset. Copy it into your working directory and print
the matching example command with:
```
layup demo prepare orbitfit
layup demo howto orbitfit
```
`prepare` writes `holman_data_working.csv` — 4135 astrometric observations of
asteroid (3666) Holman, in ADES CSV form — to the current directory, and `howto`
prints the ready-to-run command. Fit it with:
```
layup orbitfit holman_data_working.csv ADES_csv -o demo_orbitfit_output
```
This writes the best-fit barycentric Cartesian orbit and its covariance to
`demo_orbitfit_output.csv`. Supported input formats are `MPC80col`, `ADES_csv`, `ADES_psv`,
`ADES_xml`, and `ADES_hdf5`.

Convert the result to another orbit representation (Cometary, Keplerian, …):
```
layup convert demo_orbitfit_output.csv KEP -o demo_orbit_kep
```

Predict future on-sky positions, with uncertainties, for an observatory:
```
layup predict demo_orbitfit_output.csv --days 30 --station X05 -o my_predictions
```

Every verb takes `--help` for its full set of options (engine choice, IOD
method, non-gravitational parameters, parallel workers, …):
```
layup orbitfit --help
```

### Control how many CPUs layup uses

`--num-workers` (CLI) and `num_workers=` (API) default to `-1`, meaning decide
automatically: `$LAYUP_NUM_WORKERS` if set, otherwise 1 when layup is already
running inside another worker process, otherwise the CPUs available to this
process.

Set `LAYUP_NUM_WORKERS` when layup does not own the whole machine — running it
from your own process pool, or as one of several jobs on a shared node:

```
export LAYUP_NUM_WORKERS=4
```

Otherwise each copy would size its pool to the whole machine and oversubscribe
it. This is separate from `OMP_NUM_THREADS` and friends, which control threads
within a worker rather than the number of workers.

### Use the Python API

The same load → fit → convert → predict workflow is available directly from
Python. See the worked-example notebook
[`docs/notebooks/orbit_fitting_api.ipynb`](docs/notebooks/orbit_fitting_api.ipynb)
and the full documentation at [layup.readthedocs.io](https://layup.readthedocs.io).

Note that a plain `pip install -e .` does not install Jupyter — it is in the `dev`
extra. To run the notebook locally, install with `pip install -e ".[dev]"`.
