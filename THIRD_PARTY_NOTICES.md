# Algorithm and data sources

Algorithm and implementation sources are listed below. Method-specific adaptations and numerical limitations are described in `docs/EXPERIMENT_PROTOCOL.md` and the source modules.

- Chang, Yang, Zha and Zhou: Adapting to Noise Tails in Private Linear Regression. Official source: https://github.com/JinyuanChang-Lab/DifferentiallyPrivateHuberRegression . Its GPL-3.0 license is preserved under licenses/Chang_GPL-3.0.txt. Huber initialization is a normalized-loss adaptation; it is not a line-for-line CVXPY copy.
- Chen and Chua: Differentially Private Stochastic Convex Optimization under a Quantile Loss Function (ICML 2023), https://proceedings.mlr.press/v202/chen23d.html . Algorithmic source for convolution-smoothed DP-SGD/objective perturbation, including their parameter-error analysis. Current comparator adaptations are documented in docs/EXPERIMENT_PROTOCOL.md.
- Tran, Reimherr and Slavković: Differentially Private Quantile Regression (2024), DOI 10.1007/978-3-031-69651-0_2. Comparator implementation source; their smoothed KNG contribution is separate. Current Adapted OPM uses radial rather than Gaussian objective perturbation, with replacement-adjacency calibration.
- Cai, Wang and Zhang: published 2021 sparse private regression algorithms. The comparator uses published-version peeling and response clipping.
- FRAPPE: Efficient Sparse Least Absolute Deviation Regression with Differential Privacy, IEEE TIFS 19 (2024), DOI 10.1109/TIFS.2023.3349054; numerical initializer based on Su–Wang (2021), DOI 10.1016/j.cam.2021.113462. The implementation uses a finite number of ADMM iterations; see the source for numerical limitations.
- California Housing and Communities and Crime are public research datasets included under `data/`. Crime column names and prespecified exclusions are recorded in `data/crime_schema.json`.

This notice does not assert a new license for the original research manuscript, supplied datasets, or all third-party code. Retain applicable upstream notices when redistributing.
