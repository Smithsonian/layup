Controlling parallelism
========================================================================================

``--num-workers`` (command line) and ``num_workers=`` (Python API) default to ``-1``,
meaning decide automatically:

1. ``$LAYUP_NUM_WORKERS``, if set.
2. Otherwise ``1``, when layup is already running inside another worker process.
3. Otherwise the CPUs available to this process (the affinity mask on Linux, so
   ``taskset`` and cgroup limits are respected).

Set ``LAYUP_NUM_WORKERS`` when layup does not own the whole machine — for example when
running it from your own process pool, or as one of several jobs on a shared node:

.. code-block:: console

   >> export LAYUP_NUM_WORKERS=4

Without it, each copy sizes its pool to the whole machine and oversubscribes it.

.. note::

   This is separate from ``OMP_NUM_THREADS`` and the other threadpool variables, which
   control the number of threads *within* a worker rather than the number of workers.
