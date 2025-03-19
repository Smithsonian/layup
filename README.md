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
You can download the source code with:
```
git clone --recursive https://github.com/Smithsonian/layup.git
```

If you cloned the repository without `--recursive` flag, you can run
```
git submodule update --init
```
to download the required submodules, `assist`, `eigen`, and `rebound`.

Next, run
```
pip install -e .
```
to create an editable install of `layup`. If you're doing development work, you can install with
```
pip install -e ".[dev]"
```
to install all of the development packages as well.

### Linux
If you're running `layup` on a linux distribution, you should add the `layup` root directory to your`$LD_LIBRARY_PATH`, with something like
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/layup>
```
so that python's dynamic linker can find the `assist` and `rebound` library objects at runtime.

### Adding new submodule 
Note that to get the new submodules added in an existing copy of the repo you want to run
```
git submodule update --init
```
And in subsequent clones of the repo you want to run
```
git clone --recursive https://github.com/Smithsonian/layup.git
```
