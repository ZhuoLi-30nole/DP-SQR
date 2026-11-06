"""Reproduce manuscript figures from archived per-fit measurements.

Figures 2 and 4 keep means and one SE. Boxplots use the original observations
and 1.5-IQR boxplots. The paired display variants differ only by the documented
method filter or axis scale. PDF, SVG, PNG, and plotted source rows are exported
to outputs/figures by default.
"""
import argparse
from pathlib import Path
import sys

from run import ROOT

sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from dpsqr import reporting as plot

SOURCE = ROOT / "paper_results" / "plot_sources"
OUTPUT = ROOT / "outputs" / "figures"
PALETTE = {
    "sqr_gd": "#1765a6",
    "sqr_sr": "#1765a6",
    "sqr_iht": "#d55e00",
    "sqr_gd_np": "#595959",
    "sqr_sr_np": "#595959",
    "sqr_iht_np": "#999999",
    "tran_dpsgd": "#a33c76",
    "tran_opm": "#78448e",
    "huber_gd": "#007e65",
    "huber_iht": "#007e65",
    "cai_iht": "#765290",
    "frappe": "#9a7400",
}
HATCHES = {"sqr_gd_np": "//", "sqr_sr_np": "//", "sqr_iht_np": "xx"}


def box_style(method):
    color = plot.COLORS[method]
    rgb = np.array(plot.matplotlib.colors.to_rgb(color))
    return {
        "facecolor": 0.78 * rgb + 0.22,
        "edgecolor": color,
        "hatch": HATCHES.get(method),
    }


def box_legend(fig, methods):
    handles = [plot.Patch(label=plot.LABELS[method], **box_style(method))
               for method in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(4, len(handles)), columnspacing=1.1, handlelength=1.5)


def read_source(name, metric):
    path = SOURCE / f"{name}_source.csv.gz"
    frame = pd.read_csv(path)
    if frame.empty or not frame.status.eq("ok").all() or frame.diagnostic_only.any():
        raise ValueError(f"{path.name}: expected successful formal fits")
    if frame.task_id.duplicated().any() or not np.isfinite(frame[metric]).all():
        raise ValueError(f"{path.name}: duplicate fits or nonfinite {metric}")
    group = ["n_train", "p", "true_sparsity", "noise", "tau", "epsilon", "method"]
    group = [column for column in group if column in frame]
    sizes = frame.groupby(group, dropna=False).size()
    if not sizes.eq(300).all():
        raise ValueError(f"{path.name}: expected 300 fits per setting and method")
    return frame


def draw_boxes(ax, frame, metric, group, groups, methods, point_rng):
    width = 0.82 / len(methods)
    span = np.ptp(frame[metric].to_numpy())
    for method_index, method in enumerate(methods):
        color = plot.COLORS[method]
        for group_index, key in enumerate(groups):
            values = frame.loc[
                frame.method.eq(method) & frame[group].eq(key), metric
            ].to_numpy()
            if not values.size:
                continue
            x = group_index - 0.41 + (method_index + 0.5) * width
            compact = False
            if point_rng is not None and span > 0:
                q1, q3 = np.quantile(values, [0.25, 0.75])
                compact = q3 - q1 < 0.01 * span
            if compact:
                jitter = point_rng.uniform(-0.27, 0.27, size=len(values)) * width
                ax.scatter(x + jitter, values, s=3, color=color, alpha=0.12,
                           linewidths=0, rasterized=True, zorder=1)
            artists = ax.boxplot(
                [values], positions=[x], widths=0.8 * width,
                patch_artist=True, manage_ticks=False, whis=1.5,
                showfliers=not compact,
                boxprops={"edgecolor": color, "linewidth": 0.75},
                medianprops={"color": color, "linewidth": 1.35},
                whiskerprops={"color": color, "linewidth": 0.65},
                capprops={"color": color, "linewidth": 0.65},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.65,
                            "markeredgecolor": color, "markerfacecolor": color},
            )
            artists["boxes"][0].set(**box_style(method))
    ax.set_xticks(range(len(groups)), [f"{key:g}" for key in groups])
    ax.set_xlim(-0.55, len(groups) - 0.45)
    if group == "tau":
        for boundary in range(len(groups) - 1):
            ax.axvline(boundary + 0.5, color="#9e9e9e", linestyle="--",
                       linewidth=0.7, zorder=0.5)
    ax.grid(axis="y", color="#dedede", linewidth=0.4)
    ax.set_axisbelow(True)


def figure3(frame, output, point_rng):
    """Use one y scale per error family across the two privacy budgets."""
    fig, axes = plot.plt.subplots(2, 3, figsize=(7.2, 5.1), sharey="col")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.81,
                        wspace=0.35, hspace=0.55)
    methods = plot.ordered(frame)
    box_legend(fig, methods)
    panels = []
    for row, epsilon in enumerate([0.3, 0.9]):
        for col, noise in enumerate(plot.NOISES):
            ax = axes[row, col]
            data = plot.select(frame, epsilon=epsilon, noise=noise)
            draw_boxes(ax, data, "l2_error", "tau", [0.25, 0.5, 0.75],
                       methods, point_rng)
            plot.decorate(ax, row * 3 + col,
                          f"{plot.NOISE_LABELS[noise]}  |  $\\epsilon={epsilon:g}$")
            ax.set_xlabel(r"Quantile level $\tau$")
            if col == 0:
                ax.set_ylabel("Parameter L2 error")
            panels.append(dict(noise=noise, epsilon=epsilon, metric="l2_error"))
    plot.save(fig, output, "figure3_quantiles", frame, panels,
              "Parameter error across quantiles at n_train=1000 and p=10. "
              "Each error family shares a vertical scale across budgets.",
              diagnostic=False)


def figure5(frame, output, epsilon, point_rng):
    fig, axes = plot.plt.subplots(2, 3, figsize=(7.2, 5.1), sharey="row")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.83,
                        wspace=0.28, hspace=0.55)
    methods = plot.ordered(frame, high=True)
    box_legend(fig, methods)
    panels = []
    for col, noise in enumerate(["gaussian", "t2.25", "cauchy"]):
        data = plot.select(frame, noise=noise)
        for row, metric in enumerate(["l2_error", "recall"]):
            ax = axes[row, col]
            draw_boxes(ax, data, metric, "tau", [0.25, 0.5, 0.75],
                       methods, point_rng)
            plot.decorate(ax, row * 3 + col, plot.NOISE_LABELS[noise])
            ax.set_xlabel(r"Quantile level $\tau$")
            if col == 0:
                ax.set_ylabel("Parameter L2 error" if row == 0 else "Recall")
            if metric == "recall":
                ax.set_ylim(-0.03, 1.03)
            panels.append(dict(noise=noise, epsilon=epsilon, metric=metric))
    plot.save(fig, output, f"figure5_robustness_eps{epsilon:g}", frame, panels,
              f"High-dimensional parameter error and Recall at epsilon={epsilon:g}.",
              diagnostic=False)


def application_grid(frame, output, dataset, scale, point_rng, *, name=None):
    """Use identical method styling for the housing and community panels."""
    if scale == "log" and not frame.ql_tau.gt(0).all():
        raise ValueError("Figure 7 log scale requires strictly positive check loss")
    high = dataset == "community"
    fig, axes = plot.plt.subplots(3, 3, figsize=(7.2, 7.0), sharey="row")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.86,
                        wspace=0.24, hspace=0.50)
    methods = plot.ordered(frame, high=high)
    box_legend(fig, methods)
    groups = sorted(frame.n_train.unique())
    panels = []
    for row, tau in enumerate([0.25, 0.5, 0.75]):
        for col, epsilon in enumerate([0.3, 0.5, 0.9]):
            ax = axes[row, col]
            data = plot.select(frame, tau=tau, epsilon=epsilon)
            ax.set_yscale(scale)
            draw_boxes(ax, data, "ql_tau", "n_train", groups,
                       plot.ordered(data, high=high), point_rng)
            plot.decorate(ax, row * 3 + col,
                          rf"$\tau={tau:g},\ \epsilon={epsilon:g}$")
            if col == 0:
                ax.set_ylabel("Test check loss")
            if row == 2:
                ax.set_xlabel("Training sample size")
            panels.append(dict(tau=tau, epsilon=epsilon, yscale=scale))
    if name is None:
        name = f"figure7_community_{scale}" if high else "figure6_housing"
    units = "original crime-rate units" if high else "log house-value units"
    plot.save(fig, output, name, frame, panels,
              f"Test check loss in {units}. The {scale} "
              "axis is shared across the three budgets within each quantile.",
              diagnostic=False)


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="Directory containing compact *_source.csv.gz files")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--compact-points", action="store_true",
                        help="Overlay all observations for boxes with very small IQR")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == SOURCE.resolve() or SOURCE.resolve() in output.parents:
        parser.error("--output must be outside the archived plot-source directory")

    overview = read_source("sim_low_overview", "l2_error")
    quantiles = pd.concat([
        read_source(f"sim_low_quantiles_eps{epsilon}", "l2_error")
        for epsilon in (0.3, 0.9)
    ]).drop_duplicates(subset="task_id")
    crime = read_source("real_crime_quantile_grid", "ql_tau")
    housing = read_source("real_housing_quantile_grid", "ql_tau")
    high_core = read_source("sim_high_core", "l2_error")
    high_quantiles = {epsilon: read_source(f"sim_high_quantiles_eps{epsilon:g}",
                                          "l2_error") for epsilon in (0.75, 1.0)}
    supplementary = pd.concat([
        read_source("sim_low_dimensions", "l2_error"),
        read_source("sim_high_support_size", "l2_error"),
    ], ignore_index=True)

    def point_generator():
        return np.random.default_rng(20261104) if args.compact_points else None

    plot.setup()
    plot.COLORS.update(PALETTE)
    for variant, include_frappe in [("full", True), ("without_frappe", False)]:
        folder = output / variant
        folder.mkdir(parents=True, exist_ok=True)
        low = overview if include_frappe else overview.loc[overview.method.ne("frappe")]
        boxes = quantiles if include_frappe else quantiles.loc[quantiles.method.ne("frappe")]
        plot.low_overview(low, folder, diagnostic=False)
        figure3(boxes, folder, point_generator())
        suffix = "with_frappe" if include_frappe else "no_frappe"
        for file in folder.iterdir():
            name = file.name.replace("sim_low_overview", f"fig2_{suffix}")
            name = name.replace("figure3_quantiles", f"fig3_{suffix}")
            file.replace(output / name)
        folder.rmdir()
        print(f"{variant}: Figure 2 {len(low)}/{len(overview)} rows, "
              f"Figure 3 {len(boxes)}/{len(quantiles)} rows")

    output.mkdir(parents=True, exist_ok=True)
    plot.high_core(high_core, output, diagnostic=False)
    for epsilon, frame in high_quantiles.items():
        figure5(frame, output, epsilon, point_generator())
    application_grid(housing, output, "housing", "linear", point_generator())
    application_grid(housing.loc[housing.method.ne("frappe")], output,
                     "housing", "linear", point_generator(),
                     name="figure6_housing_no_frappe")
    for scale in ("log",):
        application_grid(crime, output, "community", scale, point_generator())
    plot.supplements(supplementary, output, diagnostic=False)
    print(f"Figure 7: {len(crime)} rows in both variants")
    print(f"Saved manuscript figures to {output}")


if __name__ == "__main__":
    main()
