import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb


DEFAULT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.linestyle": "-",
    "grid.linewidth": 2,
    "grid.color": "#F3F3F3",
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "axes.facecolor": "#FFF",
    "axes.edgecolor": "#333",
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "xtick.bottom": True,
    "xtick.color": "#333",
    "ytick.labelsize": 16,
    "ytick.left": True,
    "ytick.color": "#333",
    "lines.linewidth": 3,
}

PATH = Path(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH_PDF = PATH / "figures"
SAVE_PATH_PNG = PATH / "output"
SAVE_PATH_CSV = PATH / "csv"
COLORS = [
    "tab:blue",
    "tab:gray",
    "black",
    "tab:purple",
    "tab:red",
    "tab:orange",
    "tab:brown",
    "tab:olive",
    "xkcd:gold",
    "xkcd:mint",
    "#abacab",
]

ALGO_TO_LABEL = {
    "Demo3": "Ours",
    "MoDem2": "no learned reward",
    "TD-MPC2": "TD-MPC2",
    "MoDem": "MoDem",
    "LaNE": "LaNE",
}
ALGO_TO_COLOR = {
    "Ours": 1,
    "no learned reward": 0,
    "TD-MPC2": 4,
    "MoDem": 5,
    "LaNE": 6,
}


def set_style():
    matplotlib.use("agg")
    sns.set_context(context="paper", rc=DEFAULT_RC)
    sns.set_theme(style="whitegrid", rc=DEFAULT_RC)
    sns.set_palette(sns.color_palette(COLORS))


def get_runs(entity, project, tasks=None, verbose=True, **kwargs):
    api = wandb.Api(timeout=60)
    if tasks:
        kwargs["filters"] = {"$or": [{"config.task": task} for task in tasks]}
    runs = api.runs(os.path.join(entity, project), **kwargs)
    if verbose:
        print(f"Found {len(runs)} runs")
    return runs


def filter_runs(runs, verbose=True, **kwargs):
    for k, v in kwargs.items():
        runs = [
            run
            for run in runs
            if (
                run.config.get(k) in v
                if isinstance(v, (list, tuple, dict, range))
                else run.config.get(k) == v
            )
        ]
    if verbose:
        print(f"Returning {len(runs)} runs after applying filter {kwargs}")
    return runs


def get_results(fp):
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f"Error reading {fp}: {e}")
        return None
    try:
        df["step"] = df["step"] / 1e3
    except:
        pass
    return df


def save_fig(fn, dpi=300):
    fp_pdf = SAVE_PATH_PDF / fn
    fp_png = SAVE_PATH_PNG / fn
    fp_pdf.parent.mkdir(parents=True, exist_ok=True)
    fp_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fp_pdf) + ".pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(str(fp_png) + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved figure {fn}")
