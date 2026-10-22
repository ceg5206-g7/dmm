"""Static figures for modeled layout selection, cost boundaries and payload budgets."""
from itertools import product


def plot_model(rows, phases, choices, config, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    colors = ["#3b82f6", "#16a34a", "#a855f7", "#f59e0b", "#d1d5db"]
    labels = ["1D ring", "Cannon", "2.5D", "tie", "infeasible"]
    ps, gammas = sorted({c["P"] for c in config.cases}), config.gammas
    columns = list(product(config.lambdas, config.budgets))
    figure = Figure(figsize=(4 * len(columns), 2.8 * len(config.sizes)), layout="constrained")
    FigureCanvasAgg(figure)
    available = {case["layout"] for case in config.cases}
    figure.suptitle("Predicted end-to-end layout under BSP assumptions\nCandidates: " + ", ".join(sorted(available)))
    axes = figure.subplots(len(config.sizes), len(columns), squeeze=False)
    index = {(r["n"], r["P"], r["gamma"], r["lambda"], r["mu"]): r for r in choices}
    for row, n in enumerate(config.sizes):
        for col, (latency, mu) in enumerate(columns):
            ax, cells, names = axes[row][col], [], []
            for P in ps:
                cell, text = [], []
                for gamma in gammas:
                    best = index[n, P, gamma, latency, mu]["best"]
                    code = (4 if best == "infeasible" else 3 if "|" in best else
                            0 if best.startswith("ring(") else 1 if best.startswith("cannon(") else 2)
                    cell.append(code)
                    text.append(["1D", "2D", "2.5D", "tie", "×"][code])
                cells.append(cell)
                names.append(text)
            ax.imshow(cells, cmap=ListedColormap(colors), vmin=0, vmax=4, aspect="auto")
            for y, P in enumerate(ps):
                for x, gamma in enumerate(gammas):
                    ax.text(x, y, names[y][x], ha="center", va="center", fontsize=8)
            ax.set(xticks=range(len(gammas)), xticklabels=gammas, yticks=range(len(ps)), yticklabels=ps,
                   xlabel="gamma (FLOPs/word)", ylabel="logical P",
                   title=f"n={n}, lambda={latency}, mu={mu}")
    legend_indices = [i for i, name in enumerate(("ring", "cannon", "2.5d")) if name in available] + [3, 4]
    figure.legend([Patch(color=colors[i]) for i in legend_indices], [labels[i] for i in legend_indices],
                  loc="outside lower center", ncols=len(legend_indices))
    figure.savefig(output / "layout_selection.png", dpi=150)

    # One explicitly labelled representative slice; CSVs retain every scenario.
    n, gamma, latency = max(config.sizes), max(config.gammas), max(config.lambdas)
    sample = [r for r in rows if (r["n"], r["gamma"], r["lambda"]) == (n, gamma, latency)]
    native = {(r["layout"], r["P"], r["c"]): r for r in phases
              if r["phase"] == "native" and (r["n"], r["gamma"], r["lambda"]) == (n, gamma, latency)}
    names = [f"{r['layout']}\nP={r['P']}, c={r['c']}" for r in sample]
    xs = list(range(len(sample)))
    title = f"Predicted cost: n={n}, gamma={gamma}, lambda={latency}"

    def canvas(ylabel, heading=title):
        figure = Figure(figsize=(max(7, 1.2 * len(sample)), 5), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.set(xticks=xs, xticklabels=names, ylabel=ylabel, title=heading)
        ax.grid(axis="y", alpha=0.2)
        return figure, ax

    figure, ax = canvas("T_BSP / w0 (FLOP-equivalents)")
    bottom = [0.0] * len(sample)
    for field, scale, label in (("W_crit", 1, "computation"), ("C_crit", gamma, "communication"),
                                ("S", latency, "synchronization")):
        values = [scale * r[field] for r in sample]
        ax.bar(xs, values, bottom=bottom, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.legend()
    figure.savefig(output / "cost_breakdown.png", dpi=150)

    body = [native[r["layout"], r["P"], r["c"]]["T_model"] for r in sample]
    full = [r["T_model"] for r in sample]
    figure, ax = canvas("T_BSP / w0 (FLOP-equivalents)")
    ax.bar(xs, body, label="native body (including skew/replication/reduction)")
    ax.bar(xs, [a - b for a, b in zip(full, body)], bottom=body, label="common input/output conversion")
    ax.legend(fontsize=8)
    figure.savefig(output / "conversion_breakdown.png", dpi=150)

    figure, ax = canvas("T_BSP / w0 (FLOP-equivalents)")
    ax.bar([x - 0.2 for x in xs], body, width=0.4, label="native body")
    ax.bar([x + 0.2 for x in xs], full, width=0.4, label="common end-to-end")
    ax.legend()
    figure.savefig(output / "native_vs_common.png", dpi=150)

    figure, ax = canvas("peak payload / (n²/P) words", f"Logical-node payload memory: n={n}")
    ratios = [r["peak_words"] * r["P"] / n**2 for r in sample]
    ax.bar(xs, ratios, color=["#dc2626" if ratio > max(config.budgets) else "#2563eb" for ratio in ratios])
    for i, mu in enumerate(config.budgets):
        ax.axhline(mu, linestyle="--", color=colors[(i + 2) % len(colors)], label=f"M_cap = {mu} n²/P")
    for x, ratio, r in zip(xs, ratios, sample):
        ax.annotate(f"{r['peak_words']} words", (x, ratio), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.margins(y=0.15)
    ax.legend()
    figure.savefig(output / "peak_memory.png", dpi=150)
