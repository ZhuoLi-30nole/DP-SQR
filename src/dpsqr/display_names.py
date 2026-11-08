"""Publication labels; machine identifiers and archived measurements stay stable."""
METHOD_LABELS = {
    "sqr_gd": "DP-SQR-GD", "sqr_gd_np": "nonDP-SQR-GD",
    "sqr_sr": "DP-SQR-SR", "sqr_sr_np": "nonDP-SQR-SR",
    "sqr_iht": "DP-SQR-IHT", "sqr_iht_np": "nonDP-SQR-IHT",
    "tran_dpsgd": "DP-SGD", "tran_opm": "OPM",
    "huber_gd": "DP-Huber-GD", "huber_iht": "DP-Huber-IHT",
    "cai_iht": "Cai-IHT", "frappe": "FRAPPE", "lad": "LAD", "ols": "OLS",
}
BASELINE_LABELS = {"tran_dpsgd": "DP-SGD", "tran_opm": "OPM"}


def publication_frame(frame):
    """Label a copy for publication; never change a machine-readable frame."""
    result = frame.copy()
    for column in ("method", "first", "reference"):
        if column in result.columns:
            result[column] = result[column].replace(METHOD_LABELS)
    return result.rename(columns=METHOD_LABELS, index=METHOD_LABELS)


def relabel_tex(text):
    """Relabel human-readable exports while retaining values and provenance."""
    for old, new in (("Tran-SGD", "DP-SGD"), ("Tran-OPM", "OPM"),
                     (r"tran\_dpsgd", "DP-SGD"), (r"tran\_opm", "OPM"),
                     ("tran_dpsgd", "DP-SGD"), ("tran_opm", "OPM")):
        text = text.replace(old, new)
    return text
