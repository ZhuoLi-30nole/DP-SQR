"""Publication labels; machine identifiers and archived measurements stay stable."""
import re

METHOD_LABELS = {
    "sqr_gd": "DP-SQR-GD", "sqr_gd_np": "nonDP-SQR-GD",
    "sqr_sr": "DP-SQR-SR", "sqr_sr_np": "nonDP-SQR-SR",
    "sqr_iht": "DP-SQR-IHT", "sqr_iht_np": "nonDP-SQR-IHT",
    "tran_dpsgd": "Adapted DP-SGD", "tran_opm": "Adapted OPM",
    "huber_gd": "DP-Huber-GD", "huber_iht": "DP-Huber-IHT",
    "cai_iht": "Cai-IHT", "frappe": "FRAPPE", "lad": "LAD", "ols": "OLS",
}
BASELINE_LABELS = {"tran_dpsgd": "Adapted DP-SGD", "tran_opm": "Adapted OPM"}


def publication_frame(frame):
    """Label a copy for publication; never change a machine-readable frame."""
    result = frame.copy()
    for column in ("method", "first", "reference"):
        if column in result.columns:
            result[column] = result[column].replace(METHOD_LABELS)
    return result.rename(columns=METHOD_LABELS, index=METHOD_LABELS)


def relabel_tex(text):
    """Relabel publication exports idempotently; preserve numerical content."""
    aliases = (
        (("DP-SGD", "Tran-SGD", r"tran\_dpsgd", "tran_dpsgd"), "Adapted DP-SGD"),
        (("OPM", "Tran-OPM", r"tran\_opm", "tran_opm"), "Adapted OPM"),
    )
    for names, label in aliases:
        pattern = (r"(?<![\w-])(?:Adapted[ \t]+)?(?:"
                   + "|".join(re.escape(name) for name in names)
                   + r")(?![\w-])")
        text = re.sub(pattern, label, text)
    return text
