Developer guide
===============

This page is for people working *on* Layup. If you only want to use it, start
at the :doc:`home page <index>` instead.

Setting up an environment
-------------------------

Before installing any dependencies or writing code, create a virtual
environment. LINCC-Frameworks engineers primarily use ``conda``:

.. code-block:: console

   >> conda create -n <env_name> python=3.11
   >> conda activate <env_name>

Alternatively, use Python's ``venv`` module:

.. code-block:: console

   >> python -m venv venv
   >> source venv/bin/activate

Once you have an environment, install the project for local development:

.. code-block:: console

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc

Notes:

1. The single quotes around ``'[dev]'`` may not be required for your operating
   system.
2. ``pre-commit install`` initializes pre-commit for this local repository, so
   that a set of checks runs before each commit completes. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3. Installing ``pandoc`` lets you verify that automatic rendering of Jupyter
   notebooks into documentation works as expected. See the Python Project
   Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.

Running the tests
-----------------

.. code-block:: console

   >> pytest

Building the documentation
--------------------------

The Sphinx dependencies are listed in ``docs/requirements.txt`` rather than in
the ``dev`` extra, so install them separately:

.. code-block:: console

   >> pip install -r docs/requirements.txt
   >> cd docs
   >> make html

The rendered pages land in ``docs/_build/html``.
