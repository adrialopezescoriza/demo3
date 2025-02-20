import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from results import *

STEPS = 500_192
INTERVAL = 1000

SEEDS = [0, 1, 2, 3, 4]
MANISKILL_TASKS = [
    "stack-cube-semi",
    "peg-insertion-semi",
    "lift-peg-upright-semi",
    "poke-cube-semi",
    "pick-place-semi",
    "humanoid-place-apple-semi",
    "humanoid-transport-box-semi",
]
METAWORLD_TASKS = [
    "mw-assembly-semi",
    "mw-peg-insert-side-semi",
    "mw-stick-push-semi",
    "mw-stick-pull-semi",
    "mw-pick-place-semi",
]
ROBOSUITE_TASKS = [
    "robosuite-door-semi",
    "robosuite-stack-semi",
    "robosuite-pick-place-can-semi",
    "robosuite-lift-semi",
]
MANISKILL_STAGES_TASKS = [
    "stack-cube",
    "stack-cube-2-stages-semi",
    "stack-cube-1-stages-semi",
    "peg-insertion",
    "peg-insertion-2-stages-semi",
    "peg-insertion-1-stages-semi",
]
OBS = "rgbd"
NUM_ENVS = 1
ENTITY = "wandb_username"  # wandb_username
ALGORITHMS = ["DEMO3", "MoDem"]
PROJECT = "maniskill3"
TASKS_DICT = {
    "maniskill3": MANISKILL_TASKS,
    "metaworld": METAWORLD_TASKS,
    "robosuite": ROBOSUITE_TASKS,
}
TASKS = TASKS_DICT[PROJECT]


def interpolate_steps(df, step_col="step", metric_col="success", interval=INTERVAL):
    """Interpolate data for every `interval` steps."""
    if len(df) == 0:
        return df
    min_step, max_step = df[step_col].min(), df[step_col].max()
    steps = np.arange(0, max_step + 1, interval)
    interpolated_df = (
        df.set_index(step_col)
        .reindex(steps)
        .interpolate(method="linear")
        .reset_index()
        .rename(columns={"index": step_col})
    )
    interpolated_df[step_col] = interpolated_df[step_col].astype(int)
    return interpolated_df


def get_avg_df(runs, group, key, task, algorithm):
    if len(runs) == 0:
        return None
    new_key = key.replace("episode_", "")
    # new_key = key.replace('stage_1_', '')
    data = dict()
    for run in runs:
        df = run.history(keys=[f"{group}/{key}"], x_axis=f"{group}/step", pandas=True)
        df = df.rename(columns={f"{group}/{key}": new_key})
        objects = run.config.get("n_demos")
        seed = run.config.get("seed")
        df["seed"] = seed
        if len(df) == 0:
            continue
        if objects not in data:
            data[objects] = dict()
        if seed in SEEDS and (
            seed not in data[objects] or len(df) > len(data[objects][seed])
        ):
            data[objects][seed] = df

    for objects in data.keys():
        data[objects] = pd.concat(
            [
                interpolate_steps(d, step_col=f"{group}/step", metric_col=new_key)
                for d in data[objects].values()
            ],
            ignore_index=True,
        )

        # limit to plot range
        data[objects] = data[objects][data[objects][f"{group}/step"] <= STEPS]

        # clean up
        data[objects] = data[objects].rename(columns={f"{group}/step": "step"})
        if group == "train":  # average over steps
            data[objects] = data[objects].groupby(["step", "seed"]).mean().reset_index()
        data[objects][new_key] = data[objects][new_key].round(4)

        # warn if missing seeds or steps
        for seed in SEEDS:
            if seed not in data[objects]["seed"].values:
                print(f"WARNING: missing seed {seed} for {objects}")
            elif (
                len(data[objects][data[objects]["seed"] == seed])
                < len(data[objects]) / len(SEEDS) - 1
            ):
                print(f"WARNING: missing steps for seed {seed} for {objects}")

    # save to csv
    fp = SAVE_PATH_CSV / ALGO_TO_LABEL[algorithm] / f"{task}.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    # convert data to df and save to csv
    df = pd.DataFrame()
    for objects in data.keys():
        data[objects]["n_demos"] = objects
        df = pd.concat([df, data[objects]], ignore_index=True)
    df = df[["n_demos", "seed", "step", "success"]]
    _df = df[df["step"] != STEPS]
    if len(_df) > 0:
        print("WARNING: missing steps for objects:\n", _df)
    df.to_csv(fp, index=False)

    print("Average success rate:", float(df["success"].mean()))


def results_to_csv(group="eval", key="episode_success"):  # stage_1_success
    runs = get_runs(entity=ENTITY, project=PROJECT)

    for task in TASKS:
        for algo in ALGORITHMS:
            runs_ = filter_runs(
                runs, task=task, algorithm=algo, obs=OBS, num_envs=NUM_ENVS
            )
            get_avg_df(runs_, group, key, task, algo)


if __name__ == "__main__":
    results_to_csv()
