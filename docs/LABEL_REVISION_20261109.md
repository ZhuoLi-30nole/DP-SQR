# Adapted comparator label revision (2026-11-09)

The implemented quantile comparators are now displayed as **Adapted DP-SGD** and **Adapted OPM**. Their stable identifiers remain `tran_dpsgd` and `tran_opm`.

Only publication-name mapping, related documentation, label assertions, reference figure legends, and reference table labels were changed. All archived individual-fit measurements, configurations, algorithm implementations, and the experiment runner are byte-identical to the base commit. No full experiments were rerun.

Seven of the twelve manuscript reference PDFs contain these comparators and have updated legends. Five other reference PDFs are unchanged. The label-only figure package was checked against the manuscript PDFs: the seven changed files differ only in the top legend region (including automatic legend repositioning), and plotted curves, boxes, outliers, error bars, axes, and panel layout are unchanged. Numeric reference-table content is unchanged.

All 20 tests passed, including idempotent publication-name conversion and preservation of machine-readable identifiers. Archived reconstruction and numeric-table verification are recorded in `validation.json` under `label_revision_20261109`.

Original experiment source digests remain historical provenance. The new release source digest changes because the display-name module changed; it does not indicate an algorithm change. Reference hashes and release inventory were refreshed for the revised fixtures.
