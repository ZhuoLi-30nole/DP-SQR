# Algorithm and data sources

The copied numerical implementations originate from the user-supplied reproducibility project; release source hashes and archived-result provenance are recorded in paper_results/manifest.json and RELEASE_MANIFEST.json. They retain their method-specific comments and research-approximation limitations.

- Chang, Yang, Zha and Zhou: Adapting to Noise Tails in Private Linear Regression. Official source: https://github.com/JinyuanChang-Lab/DifferentiallyPrivateHuberRegression . Its GPL-3.0 license is preserved under licenses/Chang_GPL-3.0.txt. Huber initialization is a normalized-loss adaptation; it is not a line-for-line CVXPY copy.
- Tran, Reimherr and Slavković: Differentially Private Quantile Regression (2024), DOI 10.1007/978-3-031-69651-0_2. The supplied implementation and its explicit deviations are retained.
- Cai, Wang and Zhang: published 2021 sparse private regression algorithms. The supplied published-version peeling and response-clipping implementation is retained.
- FRAPPE: Efficient Sparse Least Absolute Deviation Regression with Differential Privacy, IEEE TIFS 19 (2024), DOI 10.1109/TIFS.2023.3349054; numerical initializer based on Su–Wang (2021), DOI 10.1016/j.cam.2021.113462. The supplied finite-ADMM limitations are retained.
- California Housing and Communities and Crime are public research datasets supplied with the original package. Crime column metadata was extracted as fixed JSON from the supplied author notebook; the notebook is not executed or redistributed as an experiment entry point.

This notice does not assert a new license for the original research manuscript, supplied datasets, or all third-party code. Retain applicable upstream notices when redistributing.
