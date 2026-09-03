# Sleipner data

This directory contains the geological surfaces, feeder location, and observed plume outlines used by the Sleipner experiments.

- `depth_surfaces/`, `feeders/`, and the 2010 outlines in `Sleipner_Plumes_Boundaries/` originate from the [Sleipner 2019 Benchmark Model](https://co2datashare.org/dataset/sleipner-2019-benchmark-model) distributed through CO2DataShare. The original BagIt manifests are retained with the 2010 outlines.
- [`../data_loader.py`](../data_loader.py) transposes and reverses the stored depth-surface arrays to the simulator's `(x, y)` orientation. It does not otherwise modify their values.
- `sleipner_2023_polygons/` contains the 2023 layer outlines used in the experiments. They were extracted from the plume interpretation presented by Martinez et al. (2026), [*Unraveling multilayer CO2 plumes using the entire wavefield: Case study from the Sleipner storage site*](https://doi.org/10.1190/int-2025-0016). These files are a project-derived interpretation of that publication and are not part of the CO2DataShare dataset.

The CO2DataShare material is governed by the [Sleipner CO2 Reference Dataset License](https://co2datashare.org/sleipner-2019-benchmark-model/static/license.pdf), not by the repository's BSD license. Users must comply with its attribution and redistribution conditions.
