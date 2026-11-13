# Group report

`report.pdf` is the CEG5206 team-project report; `report.tex` is its LaTeX source.

| Path | Content |
| --- | --- |
| `report.tex`, `report.pdf` | Report source and compiled PDF |
| `make_figures.py` | Draws every figure in `figures/` from the CSV files in `data/` |
| `figures/` | Vector figures used by the report |
| `data/` | Unmodified CSV files from the experiment runs |

Data provenance:

- `*_x86_0b8b2b8*`: main timing run of the submitted commit on an i5-13600K under WSL2 Ubuntu 24.04
  (one-minute load average 0.12–0.19 before and after each timed run, clean tree). `targeted_2.5d_checks_x86_0b8b2b8.md` lists the extra 2.5D
  correctness checks of the same run. The CSV files are byte-identical copies of files in
  `dmm_local_lowload_results_2026-11-10/` on `main` (`measure_15rep/`, `model/`, `model_direct/`).
- `*_0b8b2b8*`: executed counts of the submitted commit (model, ablation and padding sweeps), identical on
  both hosts; also in [`experiment-results-0b8b2b8`](https://github.com/ceg5206-g7/dmm/releases/tag/experiment-results-0b8b2b8).
- `*_9fc781b*`: ARM64 comparison timings from
  [`experiment-results-9fc781b`](https://github.com/ceg5206-g7/dmm/releases/tag/experiment-results-9fc781b).

Build:

```sh
python make_figures.py          # needs matplotlib
latexmk -pdf report.tex         # needs a TeX Live distribution with newtx and TikZ
```
