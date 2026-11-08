# GitHub release audit

Current synchronization: 8 November 2026. The dated records below document the original 6 November release. See [the update record](RELEASE_UPDATE_20261108.md) for the current changes and validation.

Prepared on 6 November 2026. The experiment-to-manuscript mapping follows the final `main.tex`, `supp.tex`, and their referenced figures and tables. Preparing this separate release did not modify the original LaTeX or experiment project.

## Included material

- All algorithms, initialization procedures, privacy accounting, data generation, preprocessing, and comparison methods in `src/dpsqr/`; the experiment runner and screening-stress script. Algorithm modules and the screening script are byte-for-byte copies of the originals. The only change to `run.py` is its introductory docstring's reference to the English README. The original and release source digests are recorded separately.
- Nine formal configurations and four smoke configurations, preserving the formal entry points and checkpoint/resume behavior.
- Two original public benchmark files and fixed Crime-variable metadata.
- Individual-fit metrics from the final figure sources, individual Housing coefficients, final screening-stress results, and rank diagnostics. The compressed measurement files total 16,977,190 bytes. Numeric strings are retained; no fits or measurements were fabricated or substituted during packaging.
- The 12 external experimental PDFs referenced by the manuscript and final-layout fixtures for 10 tables: nine numeric result tables and one application-design table.
- Formal configurations, completion states, and original source digests. Only the final screening run with `n_test=5000` and 300 repetitions is included.
- Existing third-party attribution and the full Chang GPL-3.0 license text. No new blanket license has been assigned on the authors' behalf.

## Material excluded from the release

| Original project material | Reason |
| --- | --- |
| `.venv/`, `.idea/`, `.runtime/`, `__pycache__/`, `.DS_Store` | Local interpreters, editor settings, or regenerable caches |
| `configs/pilot10*.json` | Ten-repetition pilots, outside the final manuscript |
| `results/pilot10*`, historical screening outputs, parallel benchmarks | Historical diagnostics or superseded study versions |
| `results/formal_300_20261101/` | Duplicate/incomplete later run directory; final figures trace to the completed `results/formal/` runs |
| Several GB of checkpoints and logs | Local resume artifacts; new fits create their own |
| Repeated CSV/SVG/PNG exports under `paper_outputs/` and `paper_outputs_v2/` | Necessary measurements and final PDF fixtures are retained; other exports are regenerated |
| Original plotting entry point, date-specific scheduler, delivery-check history | Original reporting functions remain in the numerical source; unified reproduction commands replace the historical workflow |
| Outdated READMEs, verification claims, and machine-specific paths in the initial copy inventory | Replaced with final-manuscript documentation and portable release manifests |

Excluded material remains in the original experiment project; it was not copied into this release.

## Code organization

The release `make_figures.py` derives from the original `make_figures2.py`. It accepts an input directory, reads compact compressed measurements, uses final manuscript filenames, and generates only figures used in the paper. Numerical algorithms are unchanged.

`make_tables.py` recomputes table values from individual fits. `verify_release.py` checks archive hashes and displayed manuscript values. `export_results.py` and `reproduce.py` connect archived reconstruction and full reruns.

During English-language standardization, the initial docstring in `run.py` was updated to point to `README.md`. This changes the byte-based source digest even though execution is unchanged. The release digest is checked against `release_source_digest`; historical `source_digest` values remain unchanged in the archived provenance. An AST comparison, excluding the introductory docstring, confirms that the runner's executable Python structure is unchanged.

## Validation and scope

1. All 12 original unit/protocol tests passed. Housing and Crime data shapes were `(20640, 5)` and `(1994, 98)`.
2. Four smoke grids completed with zero failed fits. A small screening-stress study was also executed and checked.
3. All 12 figures were rebuilt from the compressed archive; their PNGs matched the original final-output PNGs pixel for pixel.
4. Nine numeric tables were recomputed and checked at the manuscript's displayed precision. The application-design table was reconstructed from configurations.
5. The original algorithm modules and numerical runner behavior were preserved. The English-language revision changes only the runner docstring and documentation, with provenance updated accordingly.
6. The complete 750,000 formal fits and 10,500 screening fits were not rerun during packaging. This validation must not be described as a successful full study rerun from scratch.
7. Verification used an existing Python 3.12.10 environment; a fresh dependency installation was not performed. Installed versions are recorded in `requirements-locked.txt`.

The release supports both reconstruction from archived measurements and rerunning all fits from the original data. Archived reconstruction has been fully checked; fresh fitting has been exercised only at diagnostic scale. Cross-platform reruns are not guaranteed to be bitwise identical.

Machine-readable results are in `validation.json`. File SHA-256 hashes and the release inventory are in the root `RELEASE_MANIFEST.json`.
