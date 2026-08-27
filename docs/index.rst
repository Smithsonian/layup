.. layup documentation main file.

.. image:: images/layup_logo.png
  :width: 410
  :alt: Layup logo
  :align: center

========================================================================================

.. This paragraph is M.J.H.'s, taken from the Layup paper's Summary. Keep the two
   in step: if the paper's summary changes, change this with it. The only edits
   made here were dropping the paper's citation keys and setting Layup in RST
   literal markup.

The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) is under
way. The LSST is expected to raise the number of known solar system objects in
the Minor Planet Center's catalogs to roughly 127,000 near-Earth objects (NEOs),
5.1 million main-belt asteroids (MBAs), 1200–2000 Centaurs, and 37,000
trans-Neptunian objects (TNOs) — a four- to nine-fold increase over the currently
known populations. The solar system community needs efficient tools to maximize
the scientific yield of the survey.

``Layup`` is an open-source package for orbit determination at LSST scale that
serves as a companion to the `Sorcha <https://sorcha.readthedocs.io>`_ survey
simulator. ``Layup`` is built on REBOUND/ASSIST for ephemeris-quality numerical
integrations, with a C++ engine and a Python command-line interface and API. It
can ingest astrometry in obs80 and ADES formats, as well as radar range and
Doppler observations, streak (position + rate) measurements from shift-and-stack
surveys, and space-based observations. ``Layup`` provides two orbit
parameterizations — a 6-parameter Cartesian state and a Bernstein-Khushalani
basis (distance-scaled parameters in a local tangent-plane reference frame) — and
can fit both gravitational and non-gravitational accelerations. Gauss and
Bernstein-Khushalani initial orbit determination (IOD) are included, and
additional IOD methods can be added easily. Every ``Layup`` fit reports a full
state covariance, which it propagates through element and frame conversions and
through ephemeris predictions to support attribution and linking.


Installation
------------

Layup builds a C++ extension, so installing it needs a compiler as well as a
recent Python.

============  ==================================================================
Python        3.11 or newer
Compiler      a C++17 compiler — Xcode command line tools on macOS,
              ``build-essential`` or equivalent on Linux
pip           21.3 or newer (editable installs need PEP 660 support)
Platforms     macOS and Linux. Windows is not supported and is not tested
Disk          about 3.2 GB free, roughly 1.5 GB of it the reference data
              fetched by ``layup bootstrap``
============  ==================================================================

.. warning::

   Do **not** install Layup with ``pip install layup``. The ``layup`` name on
   PyPI currently holds an unrelated placeholder package, so that command
   succeeds and installs nothing, with no error. Install from source as below
   until the first release is published.

It is a good idea to create a virtual environment first:

.. code-block:: console

   >> python -m venv venv
   >> source venv/bin/activate

Then clone and install:

.. code-block:: console

   >> git clone https://github.com/Smithsonian/layup.git
   >> cd layup
   >> pip install .

If that fails with *"File setup.py or setup.cfg not found"*, your pip predates
PEP 660 — run ``pip install --upgrade pip`` first. The ``python3`` shipped with
macOS is too old; install a newer Python before creating the environment.


Quickstart
----------

Layup needs SPICE planetary kernels, the small-body kernel, MPC observatory
codes, and the astrometry debiasing tables. Download them once with:

.. code-block:: console

   >> layup bootstrap

This fetches roughly 1 GB, which expands to about 1.5 GB on disk.

Fit an orbit
^^^^^^^^^^^^

Layup bundles a demo dataset. Copy it into your working directory and print the
matching example command:

.. code-block:: console

   >> layup demo prepare orbitfit
   >> layup demo howto orbitfit

``prepare`` writes ``holman_data_working.csv`` — 4135 astrometric observations
of asteroid (3666) Holman, in ADES CSV form — to the current directory, and
``howto`` prints the ready-to-run command. Fit it with:

.. code-block:: console

   >> layup orbitfit holman_data_working.csv ADES_csv -o demo_orbitfit_output

This writes the best-fit barycentric Cartesian orbit and its covariance to
``demo_orbitfit_output.csv``. Supported input formats are ``MPC80col``,
``ADES_csv``, ``ADES_psv``, ``ADES_xml``, and ``ADES_hdf5``.

Convert and predict
^^^^^^^^^^^^^^^^^^^

Convert the result to another orbit representation (Cometary, Keplerian, …):

.. code-block:: console

   >> layup convert demo_orbitfit_output.csv KEP -o demo_orbit_kep

Predict future on-sky positions, with uncertainties, for an observatory:

.. code-block:: console

   >> layup predict demo_orbitfit_output.csv --days 30 --station X05 -o my_predictions

Every verb takes ``--help`` for its full set of options — engine choice, IOD
method, non-gravitational parameters, parallel workers, and so on:

.. code-block:: console

   >> layup orbitfit --help


Where to go next
----------------

* The same load → fit → convert → predict workflow is available from Python.
  The :doc:`orbit fitting API notebook <notebooks/orbit_fitting_api>` works
  through it end to end.
* :doc:`Controlling parallelism <parallelism>` — how Layup sizes its worker
  pool, and how to stop it oversubscribing a shared machine.
* `API Reference <autoapi/index.html>`_ — every public module, class and
  function.
* :doc:`Developer guide <dev_guide>` — setting up a development environment and
  running the tests.

.. note::

   A plain install does not include Jupyter; it lives in the ``dev`` extra. To
   run the notebooks locally, install with ``pip install -e ".[dev]"``.


Citing Layup
------------

If Layup contributes to work you publish, please cite it. Citation details will
be listed here with the first release.


Getting help
------------

Please open an issue at
`github.com/Smithsonian/layup/issues <https://github.com/Smithsonian/layup/issues>`_
for bug reports, questions, and feature requests.

.. Once Smithsonian/layup#466 merges, link CONTRIBUTING.md, CODE_OF_CONDUCT.md
   and SUPPORT.md from this section -- they do not exist on main yet, so they
   are deliberately not linked.

.. toctree::
   :hidden:

   Home page <self>
   Controlling parallelism <parallelism>
   Orbit fit status flags <fit_flags>
   Notebooks <notebooks>
   API Reference <autoapi/index>
   Developer guide <dev_guide>
