MLflow Integration
==================

FCVOpt integrates with `MLflow <https://mlflow.org/>`_ to automatically track hyperparameter
optimization runs. Tracking is initialized lazily on the first call to :meth:`run` or
:meth:`optimize`, so simply creating an optimizer does not start a run.

What Gets Tracked
-----------------

Each optimization run is organized as a **parent run** with one **nested child run per
trial evaluation**. The following are tracked automatically:

**Parent run**

* Tags: framework name, acquisition function, batch acquisition mode, random seed
* Parameters: ``minimize``, ``acquisition_q``, ``n_jobs``, ``model_checkpoint_freq``
* Metrics (logged per iteration): ``f_inc_obs``, ``f_inc_est``, ``acq_val``, ``fit_time``, ``acq_opt_time``
* Artifacts:

  * ``config_space.json`` — the hyperparameter configuration space
  * ``evals/eval_NNN.json`` — one JSON file per trial evaluation (used for run restoration)
  * ``iterations/iter_NNN.json`` — per-iteration snapshots with incumbent configuration and acquisition metrics
  * ``checkpoints/iter_NNN_model_state.pth`` — GP model state dicts (frequency controlled by ``model_checkpoint_freq``)

**Child runs** (one per trial)

* Parameters: hyperparameter configuration values, trial index
* Metrics: ``loss``, ``eval_time``

Specifying Where to Store Runs
-------------------------------

Two mutually exclusive parameters control where MLflow stores run data:

* **``tracking_dir``** — a plain local directory path (e.g., ``"./my_experiments"``).
  Internally converted to a ``file:`` URI.
* **``tracking_uri``** — a fully-formed MLflow tracking URI. Use this for remote servers
  (``"http://localhost:5000"``) or explicit ``file:`` and database URIs
  (``"sqlite:///mlflow.db"``). Takes precedence over ``tracking_dir`` if both are set.

If neither is provided, runs are stored in ``./mlruns`` relative to the working directory.

.. code-block:: python

   from fcvopt.optimizers import BayesOpt  # or FCVOpt

   # Local directory (most common)
   optimizer = BayesOpt(
       obj=cv_obj,
       config=config_space,
       tracking_dir='./my_experiments',
       experiment='rf_optimization',
       run_name='run_01',
   )

   # Remote MLflow server
   optimizer = BayesOpt(
       obj=cv_obj,
       config=config_space,
       tracking_uri='http://localhost:5000',
       experiment='rf_optimization',
   )

The ``experiment`` parameter groups related runs under a named experiment in the MLflow UI
(defaults to ``"BayesOpt"`` or ``"FCVOpt"`` if not specified). The ``run_name`` parameter
names the individual run (defaults to a timestamp string).

Running the Optimizer
---------------------

:meth:`run` and :meth:`optimize` differ in how they count iterations:

* **``run(n_iter)``** — performs exactly ``n_iter`` acquisition steps, not counting
  the initial random evaluations.
* **``optimize(n_trials)``** — performs ``n_trials`` total evaluations, including the
  initial random phase.

The MLflow run is **not closed automatically** after :meth:`run` or :meth:`optimize`
returns. This allows seamless continuation runs on the same instance. Use the optimizer
as a context manager to ensure the run is properly closed:

.. code-block:: python

   with BayesOpt(obj=cv_obj, config=config_space, tracking_dir='./experiments') as bo:
       bo.run(n_iter=20, n_init=5)
       bo.run(n_iter=10)   # continuation — appends to the same MLflow run
   # MLflow run closed automatically on exit

Alternatively, call :meth:`end_run` explicitly:

.. code-block:: python

   bo = BayesOpt(obj=cv_obj, config=config_space, tracking_dir='./experiments')
   best_config = bo.optimize(n_trials=30, n_init=5)
   bo.end_run()

Restoring a Run
---------------

You can restore a previous run and continue optimization from where it left off. The
optimizer reloads all evaluated configurations and the final GP model checkpoint from
the MLflow artifact store.

Restore by **run ID** (most reliable — the run ID is printed or visible in the MLflow UI):

.. code-block:: python

   restored = BayesOpt.restore_from_mlflow(
       obj=cv_obj,
       run_id='your_run_id_here',
       tracking_dir='./experiments',
   )
   best_config = restored.optimize(n_trials=20)
   restored.end_run()

Restore by **experiment and run name** (useful when you set a meaningful ``run_name``
on the original run):

.. code-block:: python

   restored = BayesOpt.restore_from_mlflow(
       obj=cv_obj,
       experiment_name='rf_optimization',
       run_name='run_01',
       tracking_dir='./experiments',
   )
   best_config = restored.optimize(n_trials=20)
   restored.end_run()

.. note::

   Either ``run_id`` **or** both ``experiment_name`` and ``run_name`` must be provided.
   ``tracking_uri`` and ``tracking_dir`` are mutually exclusive.

MLflow UI
---------

To browse tracked experiments in the MLflow web interface, launch the UI pointing at
your tracking directory:

.. code-block:: bash

   mlflow ui --backend-store-uri ./my_experiments

Then open ``http://localhost:5000`` in your browser. The UI lets you compare runs,
plot metrics over iterations, download artifacts, and retrieve run IDs for restoration.
