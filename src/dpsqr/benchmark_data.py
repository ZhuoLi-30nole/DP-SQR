"""Raw public benchmark loading; no fitting/scaling before the train/test split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MISSING_COMMUNITY_PREDICTORS = [
    "OtherPerCap", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
    "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
    "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
    "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "PolicCars",
    "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "PolicBudgPerPop",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_housing(source: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = source / "realdata/lowdim_house_data/cadata_new.txt"
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected 9-column cadata_new.txt, got {raw.shape}")
    # Original notebook keeps income, age, population, households, rooms, in that order.
    y_log = np.log(raw[:, 0])
    y = y_log
    X = raw[:, [1, 2, 5, 6, 3]]
    return X.astype(float), y.astype(float), {
        "source_file": str(path), "source_sha256": sha256(path),
        "source_rows": int(len(y)), "predictors": ["median_income", "housing_median_age", "population", "households", "total_rooms"],
        "response": "log(median_house_value)",
        "response_transform": "natural_log",
        "preprocessing": "raw covariates; scaling is fitted later on training rows only",
    }


def load_crime(source: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = source / "realdata/highdim/Data.txt"
    schema_path = source / "crime_schema.json"
    schema = json.loads(schema_path.read_text())
    names, selected = schema["columns"], schema["selected"]
    frame = pd.read_csv(path, header=None, names=names, dtype=str)
    selected = [c for c in selected if c not in {"pctUrban", "OwnOccQrange", "RentQrange"}]
    selected = [c for c in selected if c not in MISSING_COMMUNITY_PREDICTORS]
    response = "ViolentCrimesPerPop"
    frame = frame[selected]
    frame = frame[frame[response] != "?"].reset_index(drop=True)
    # Preserve the fixed feature exclusions; scaling is deferred to the training split.
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Unexpected missing values after Chang's fixed column exclusions")
    y = numeric[response].to_numpy(float)
    X = numeric.drop(columns=response).to_numpy(float)
    feature_names = [c for c in numeric.columns if c != response]
    # Original code discards `fold` after standardization, then adds intercept in Huber.
    if feature_names and feature_names[0] == "fold":
        X, feature_names = X[:, 1:], feature_names[1:]
    if len(y) != 1994:
        raise ValueError(f"Chang community preprocessing expected n=1994, got {len(y)}")
    return X.astype(float), y.astype(float), {
        "source_file": str(path), "source_sha256": sha256(path),
        "schema_file": str(schema_path), "schema_sha256": sha256(schema_path),
        "source_rows": int(len(y)), "predictor_count": int(X.shape[1]), "predictors": feature_names,
        "response": response, "response_transform": "raw violent-crime rate; training-only z-score downstream",
        "preprocessing": "fixed exclusions; raw inputs; training-only scaling downstream; discard fold",
    }

