# Reproduction validation

## Coverage

The repository includes the algorithms and privacy accounting, nine formal experiment configurations, four small diagnostic configurations, two public benchmark inputs, individual-fit measurements, and manuscript reference figures and tables. The [manuscript-to-code map](REPRODUCIBILITY.md) lists their uses.

`make_figures.py` builds 12 experimental figures. `make_tables.py` recomputes nine numeric result tables and the application-design table. `verify_release.py` checks source and archive integrity, reference figures, manuscript table values, and publication labels.

## Verified checks

- All 19 code tests passed, covering numerical/protocol behavior, forced-coordinate validation, and display labels.
- Both benchmark datasets loaded with the expected shapes: Housing `(20640, 5)` and Crime `(1994, 98)`.
- Four small experiment grids and a small screening study completed successfully.
- Archived reconstruction of all 12 figures and 10 tables passed. Nine numeric tables matched the manuscript at the displayed precision; the application-design table matched the configurations.
- Figure source data and plotting geometry were checked against the manuscript-aligned references.

## Limits

The complete study of 750,000 formal method fits and 10,500 screening fits has not been rerun as part of repository verification. Fresh fitting has been exercised at diagnostic scale. Archived reconstruction and a full rerun from data are separate operations.

Verification used an existing Python 3.12.10 environment. A fresh dependency installation has not been tested. Installed versions are recorded in `../requirements-locked.txt`; results across operating systems and numerical libraries need not be bitwise identical.

Machine-readable validation records are in `validation.json`. File hashes and the repository inventory are in `../RELEASE_MANIFEST.json`. Archived fit provenance is retained under `../paper_results/` for traceability.
