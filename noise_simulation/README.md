# Noise Simulation Folder

This folder contains the repository-side interface to the noise-simulation stage of the thesis workflow.

Its role is to:

- convert clustered or backbone-derived tracks into Doc.29-style inputs,
- run experiment-level and ground-truth batch jobs, and
- compare aggregated noise outputs against reference results.

Main entry scripts:

- [`generate_doc29_inputs.py`](generate_doc29_inputs.py)
- [`run_doc29_experiment.py`](run_doc29_experiment.py)
- [`run_doc29_ground_truth.py`](run_doc29_ground_truth.py)
- [`compare_experiment_totals.py`](compare_experiment_totals.py)

Important limitation:

- the full simulation setup may depend on external code, external assets, or a separately maintained Doc.29 implementation that is not guaranteed to be included in a thesis-distribution copy of this repository.

Therefore, in a shared or archived version of the project, this folder may be present mainly as a structural reference.

If the simulation backend is available locally, also see:

- [`doc-29-implementation/README.md`](doc-29-implementation/README.md)
