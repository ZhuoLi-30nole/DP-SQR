# Benchmark inputs

The repository includes the following public benchmark inputs.

| File | Use |
| --- | --- |
| `data/realdata/lowdim_house_data/cadata_new.txt` | California Housing; 20,640 records; nine recorded columns |
| `data/realdata/highdim/Data.txt` | Communities and Crime raw records, including missing-value markers |
| `data/crime_schema.json` | Fixed column names and prespecified attribute selection |

The housing loader uses the log of median house value as the response and selects median income, housing age, population, households, and total rooms, in that order. The community loader applies the fixed exclusions in `src/dpsqr/benchmark_data.py`, drops the fold indicator, and retains 1,994 records with 98 predictors. An intercept is appended downstream.

Within each repetition, a seeded permutation selects the records and assigns the first 80% to training and the remainder to testing. Centering/scaling parameters are calculated only from the training observations. Housing log responses are centered; community responses are standardized. Test losses are returned to log house-value or original crime-rate units. Coefficients are transformed back to the recorded predictor units before summarizing.

Dataset citations and implementation provenance are preserved in `THIRD_PARTY_NOTICES.md`. The accompanying third-party license notice does not assign a new license to the datasets. Dataset SHA-256 values are listed in `RELEASE_MANIFEST.json`.

The `paper_results/` files contain measurements from synthetic/public-data benchmarks, including repetition identifiers, seeds, losses, and housing coefficients. They do not contain model fits on a confidential clinical or other private dataset.
