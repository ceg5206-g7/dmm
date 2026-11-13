"""Figures for the CEG5206 group report.

Every number plotted here is read from the unmodified CSV files in ``data/``:

* ``*_0b8b2b8*``  -> release ``experiment-results-0b8b2b8`` (model capture)
* ``*_9fc781b*``  -> release ``experiment-results-9fc781b`` (clean ARM64 timing run)
* ``*_x86_0b8b2b8*`` -> low-load x86/WSL2 rerun of the submitted commit

Run:  python make_figures.py   (writes PDF figures to ``figures/``)
"""
import csv
import os
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def half_up(value, digits):
    """Round half away from zero, the convention used in the report's tables."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

LAYOUT_COLOUR = {"ring": "#6C8EBF", "cannon": "#3C8D5A", "2.5d": "#D08C2C", "2.5d-direct": "#B5367A"}
LAYOUT_LABEL = {"ring": "Ring", "cannon": "Cannon", "2.5d": "2.5D", "2.5d-direct": "2.5D-direct"}
PHASES = ["input_conversion", "replication", "skew", "multiply", "reduction", "output_conversion"]
PHASE_LABEL = {"input_conversion": "input conversion", "replication": "replication", "skew": "skew",
               "multiply": "multiply", "reduction": "reduction", "output_conversion": "output conversion"}
PHASE_COLOUR = {"input_conversion": "#C9CED6", "replication": "#D08C2C", "skew": "#8FB3D9",
                "multiply": "#3C6E9F", "reduction": "#B5367A", "output_conversion": "#8A93A0"}
TEXTWIDTH = 6.3  # inches, matches the LaTeX text block


def rows(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def phase_breakdown():
    """Critical communication per phase at n=1024 (normalised by n^2/P)."""
    n = 1024
    data = defaultdict(dict)
    for r in rows("phase_model_0b8b2b8.csv"):
        if r["n"] == str(n) and r["gamma"] == "1" and r["lambda"] == "100" and r["phase"] in PHASES:
            P = int(r["P"])
            data[(P, r["layout"])][r["phase"]] = int(r["C_crit"]) / (n * n / P)
    cases = [(16, "ring"), (16, "cannon"), (64, "ring"), (64, "cannon"), (64, "2.5d"), (64, "2.5d-direct"),
             (256, "ring"), (256, "cannon"), (256, "2.5d"), (256, "2.5d-direct")]
    xmax = 70
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))
    ypos, labels = [], []
    y = 0.0
    last_P = None
    for P, layout in cases:
        if last_P is not None and P != last_P:
            y += 0.55
        last_P = P
        left = 0.0
        for ph in PHASES:
            v = data[(P, layout)].get(ph, 0.0)
            if v <= 0:
                continue
            ax.barh(y, min(v, xmax - left), left=left, height=0.72, color=PHASE_COLOUR[ph],
                    edgecolor="white", linewidth=0.5)
            left += v
        total = sum(data[(P, layout)].values())
        if total > xmax:
            ax.text(xmax - 0.6, y, r"%.0f $\rightarrow$" % total, va="center", ha="right", color="white",
                    fontsize=7.2, fontweight="bold")
        else:
            ax.text(total + 0.7, y, str(half_up(total, 1)), va="center", ha="left", fontsize=7.2)
        ypos.append(y)
        labels.append("%s, $P$=%d" % (LAYOUT_LABEL[layout], P))
        y += 1.0
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"critical communication $C_{\mathrm{crit}}$ per phase, in units of $n^2/P$ words ($n=1024$)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(color=PHASE_COLOUR[p], label=PHASE_LABEL[p]) for p in PHASES]
    ax.legend(handles=handles, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              handlelength=1.2, columnspacing=1.0, fontsize=7)
    fig.savefig(os.path.join(OUT, "phase_breakdown.pdf"))
    plt.close(fig)


def memory():
    n = 1024
    peak = {}
    for r in rows("model_0b8b2b8.csv"):
        if r["n"] == str(n) and r["gamma"] == "1" and r["lambda"] == "100":
            P = int(r["P"])
            peak[(P, r["layout"])] = int(r["peak_words"]) / (n * n / P)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.48, 2.1))
    groups = [16, 64, 256]
    layouts = ["ring", "cannon", "2.5d", "2.5d-direct"]
    width = 0.2
    for i, layout in enumerate(layouts):
        xs, ys = [], []
        for g, P in enumerate(groups):
            if (P, layout) in peak:
                xs.append(g + (i - 1.5) * width)
                ys.append(peak[(P, layout)])
        ax.bar(xs, ys, width=width * 0.92, color=LAYOUT_COLOUR[layout], label=LAYOUT_LABEL[layout])
        for x, v in zip(xs, ys):
            ax.text(x, v + 0.6, "%g" % float(half_up(v, 2)), ha="center", va="bottom", fontsize=6.3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    for mu, ls in ((8, "--"), (32, ":")):
        ax.axhline(mu, color="#444444", lw=0.8, ls=ls)
        ax.text(2.55, mu + 0.5, r"$\mu=%d$" % mu, ha="right", va="bottom", fontsize=7, color="#444444")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["$P$=16", "$P$=64", "$P$=256"])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, 36)
    ax.set_ylabel(r"peak live payload / ($n^2/P$)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handlelength=1.0,
              columnspacing=0.7, handletextpad=0.4)
    fig.savefig(os.path.join(OUT, "peak_memory.pdf"))
    plt.close(fig)


def selection():
    """Winning end-to-end layout for the mu=32 budget (mu=8 selects Cannon everywhere)."""
    sel = {}
    for r in rows("layout_selection_direct_0b8b2b8.csv"):
        if r["mu"] == "32":
            sel[(int(r["n"]), int(r["lambda"]), int(r["P"]), int(r["gamma"]))] = r["best"].split("(")[0]
    codes = {"cannon": 0, "2.5d-direct": 1, "2.5d": 2, "ring": 3}
    cmap = ListedColormap([LAYOUT_COLOUR["cannon"], LAYOUT_COLOUR["2.5d-direct"], LAYOUT_COLOUR["2.5d"],
                           LAYOUT_COLOUR["ring"]])
    Ps, gammas = [16, 64, 256], [1, 10, 100, 1000]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH * 0.5, 2.15), sharex=True, sharey=True)
    for i, lam in enumerate((100, 10000)):
        for j, n in enumerate((256, 512, 1024)):
            ax = axes[i][j]
            grid = [[codes[sel[(n, lam, P, g)]] for g in gammas] for P in Ps]
            ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
            ax.set_xticks(range(4))
            ax.set_xticklabels(["1", "10", "$10^2$", "$10^3$"], fontsize=6.3)
            ax.set_yticks(range(3))
            ax.set_yticklabels([str(P) for P in Ps], fontsize=6.3)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
            for x in range(1, 4):
                ax.axvline(x - 0.5, color="white", lw=0.6)
            for y in range(1, 3):
                ax.axhline(y - 0.5, color="white", lw=0.6)
            if i == 0:
                ax.set_title("$n$=%d" % n, fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(r"$\lambda=10^{%d}$" % (2 if lam == 100 else 4) + "\n$P$", fontsize=7.2)
    fig.supxlabel(r"$\gamma$ (FLOPs per word)", fontsize=7.5, y=0.0)
    handles = [Patch(color=LAYOUT_COLOUR["cannon"], label="Cannon"),
               Patch(color=LAYOUT_COLOUR["2.5d-direct"], label="2.5D-direct")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.07), ncol=2, frameon=False,
               fontsize=6.8)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    fig.savefig(os.path.join(OUT, "layout_selection.pdf"))
    plt.close(fig)


PLATFORMS = {"x86": ("measured_serial_x86_0b8b2b8.csv", "measured_threaded_x86_0b8b2b8.csv",
                     "x86 (0b8b2b8)", "-"),
             "arm": ("measured_serial_9fc781b.csv", "measured_threaded_9fc781b.csv", "ARM64 (9fc781b)", "--")}


def load_measurements():
    med = {}
    for plat, (serial, threaded, _, _) in PLATFORMS.items():
        for name, H in ((serial, 1), (threaded, 4)):
            for r in rows(name):
                if r["phase"] == "end_to_end":
                    med[(plat, r["layout"], int(r["n"]), H)] = {
                        "total": float(r["total_median_s"]) * 1e3,
                        "compute": float(r["compute_median_s"]) * 1e3,
                        "copy": float(r["copy_median_s"]) * 1e3,
                        "sync": float(r["synchronization_median_s"]) * 1e3,
                        "residual": float(r["relative_residual"]) * 100,
                        "speedup": float(r["speedup"]) if r["speedup"] else 1.0,
                    }
    return med


def measurement():
    """Prepared single-machine measurements on the two hosts."""
    med = load_measurements()
    sizes = [256, 512, 1024]
    markers = {"ring": "o", "cannon": "s"}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15),
                                     gridspec_kw={"width_ratios": [1, 1.25, 1]})
    for plat, (_, _, label, ls) in PLATFORMS.items():
        for layout in ("ring", "cannon"):
            a1.plot(sizes, [med[(plat, layout, n, 4)]["speedup"] for n in sizes], ls=ls,
                    marker=markers[layout], ms=3.2, lw=1.0, color=LAYOUT_COLOUR[layout],
                    label="%s, %s" % (LAYOUT_LABEL[layout], label.split()[0]))
            a3.plot(sizes, [med[(plat, layout, n, 4)]["residual"] for n in sizes], ls=ls,
                    marker=markers[layout], ms=3.2, lw=1.0, color=LAYOUT_COLOUR[layout])
    a1.axhline(4, color="#888888", lw=0.6, ls=":")
    a1.set_ylim(0, 4.5)
    a1.set_ylabel("speedup at $P$=$H$=4")
    a1.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=2.0)
    a3.axhline(0, color="#444444", lw=0.6)
    a3.set_ylim(-5, 30)
    a3.set_ylabel("model residual at $H$=4 (%)")
    for ax in (a1, a3):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("$n$")
        ax.grid(alpha=0.25, lw=0.5)
    # time components at P=H=4, as fractions of the component sum
    comp_colour = {"compute": "#3C6E9F", "copy": "#8FB3D9", "sync": "#B5367A"}
    bars = [(plat, layout, n) for n in (256, 1024) for plat in ("arm", "x86") for layout in ("ring", "cannon")]
    xs, ticks = [], []
    x = 0.0
    for i, (plat, layout, n) in enumerate(bars):
        if i and i % 4 == 0:
            x += 0.6
        elif i and i % 2 == 0:
            x += 0.25
        m = med[(plat, layout, n, 4)]
        tot = m["compute"] + m["copy"] + m["sync"]
        bottom = 0.0
        for k in ("compute", "copy", "sync"):
            a2.bar(x, m[k] / tot * 100, bottom=bottom, width=0.8, color=comp_colour[k],
                   label=k if i == 0 else None, edgecolor="white", linewidth=0.3)
            bottom += m[k] / tot * 100
        xs.append(x)
        ticks.append(("R" if layout == "ring" else "C"))
        x += 1.0
    a2.set_xticks(xs)
    a2.set_xticklabels(ticks, fontsize=6.5)
    a2.tick_params(axis="x", length=0, pad=2)
    for centre, text in ((sum(xs[0:2]) / 2, "ARM"), (sum(xs[2:4]) / 2, "x86"),
                         (sum(xs[4:6]) / 2, "ARM"), (sum(xs[6:8]) / 2, "x86")):
        a2.text(centre, -0.15, text, ha="center", va="top", fontsize=6.5, transform=a2.get_xaxis_transform())
    a2.text(sum(xs[0:4]) / 4, -0.27, "$n$=256", ha="center", va="top", fontsize=7.2,
            transform=a2.get_xaxis_transform())
    a2.text(sum(xs[4:8]) / 4, -0.27, "$n$=1024", ha="center", va="top", fontsize=7.2,
            transform=a2.get_xaxis_transform())
    a2.set_ylim(0, 100)
    a2.set_ylabel("share of $P$=4 time (%)")
    a2.legend(frameon=False, fontsize=6.2, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "measurement.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    phase_breakdown()
    memory()
    selection()
    measurement()
    print("figures written to", OUT)
