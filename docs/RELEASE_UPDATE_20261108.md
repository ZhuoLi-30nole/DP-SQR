# Final manuscript synchronization — 8 November 2026

Base public commit: `fc9a5620a95a12c8b295ea5f2501c14e65b75061` (6 November 2026). The manuscript snapshot is the finalized black-text version of 7 November 2026.

## Changes

- Use one publication-label map for plots and human-readable tables: DP-SGD and OPM. Machine IDs, configurations, task IDs, CSV method keys, and archived measurements remain unchanged.
- Validate public forced-coordinate indices before integer conversion in both SR entry points. Fractional indices, booleans, duplicate indices, and out-of-range indices are rejected. Valid integer-valued inputs retain their previous interpretation and order.
- Clarify Chen–Chua algorithmic contributions, the Tran implementation lineage, and the radial OPM adaptation. Correct obsolete supplement numbers and the reference to the absent assurance.py module.
- State the distinction between fixed-public-start statistical theorems and the budgeted private initializer used in experiments.
- Synchronize the 12 reference figures and affected table labels with the final manuscript; refresh release hashes while preserving historical source/provenance fields.
- Add seven regression tests covering input validation and publication labels. The existing twelve protocol tests remain.

## Numerical preservation and validation

The update preserves all raw benchmark inputs, archived numerical measurements, formal configurations, update equations, clipping thresholds, noise calibration, privacy allocations, and random seeds. No formal experiment was rerun to create this release.

Validation details are recorded in docs/validation.json under release_update_20261108. Historical verification results elsewhere in the documentation retain their original dates and scope.

The release source digest changes because publication helpers and input validation reside under src/. Historical digests remain evidence of the original runs; the current release digest identifies the current runnable source. Existing checkpoint fingerprints must not be rewritten to impersonate runs of the new source. Use a new output directory for any future fitting run, as required by the existing runner.

## Verification commands

```bash
python run.py check --tests
python reproduce.py --mode archived
```

The second command reconstructs figures and tables from archived measurements. It does not execute the full experimental study. The label checks supplement, rather than replace, numerical and archive-integrity verification.
