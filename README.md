<h1 align="center">
<img src="https://raw.githubusercontent.com/Smithsonian/layup/main/docs/images/layup_logo.png" width="500">
</h1><br>

# layup
Orbit fitting at LSST scale

[![ci](https://github.com/Smithsonian/layup/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/Smithsonian/layup/actions/workflows/smoke-test.yml)
[![pytest](https://github.com/Smithsonian/layup/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/Smithsonian/layup/actions/workflows/testing-and-coverage.yml)
[![Documentation Status](https://readthedocs.org/projects/layup/badge/?version=latest)](https://layup.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Version](https://img.shields.io/pypi/v/layup)](https://pypi.python.org/pypi/layup)
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

## Setup
To install layup, all its dependencies, as well as the ephemeris and reference data, you need about 3.2G of free space.
Before installing layup, it's a great idea to create a virtual environment with either `conda` or `venv`.

You can download the source code with:
```
git clone --recursive https://github.com/Smithsonian/layup.git
```

If you cloned the repository without `--recursive` flag, you can run
```
git submodule update --init
```
to download the required submodules, `assist`, `eigen`, and `rebound`.

Next, enter the layup directory and run
```
pip install -e .
```
to create an editable install of `layup`. If you're doing development work, you can install with
```
pip install -e ".[dev]"
```
to install all of the development packages as well.

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
astrometry debiasing tables). This is a one-time download of a few hundred MB:
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
layup orbitfit holman_data_working.csv ADES_csv -o my_orbit
```
This writes the best-fit barycentric Cartesian orbit and its covariance to
`my_orbit.csv`. Supported input formats are `MPC80col`, `ADES_csv`, `ADES_psv`,
`ADES_xml`, and `ADES_hdf5`.

Convert the result to another orbit representation (Cometary, Keplerian, …):
```
layup convert my_orbit.csv KEP -o my_orbit_kep
```

Predict future on-sky positions, with uncertainties, for an observatory:
```
layup predict my_orbit.csv --days 30 --station X05 -o my_predictions
```

Every verb takes `--help` for its full set of options (engine choice, IOD
method, non-gravitational parameters, parallel workers, …):
```
layup orbitfit --help
```

### Use the Python API

The same load → fit → convert → predict workflow is available directly from
Python. See the worked-example notebook
[`docs/notebooks/orbit_fitting_api.ipynb`](docs/notebooks/orbit_fitting_api.ipynb)
and the full documentation at [layup.readthedocs.io](https://layup.readthedocs.io).
