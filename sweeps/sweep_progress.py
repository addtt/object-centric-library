import argparse
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from utils.paths import ROOT

"""Call with: python -m sweeps.sweep_progress SWEEP_NAME"""

FILL_CHAR = "#"
EMPTY_CHAR = "."

if __name__ == "__main__":

    # TODO compute this later when needed, infer best number given lengths of other strings
    columns = shutil.get_terminal_size().columns
    bar_len = columns - 30

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_name", help="sweep name")
    args = parser.parse_args()
    sweep_dir = ROOT / f"outputs/sweeps/sweep_{args.sweep_name}"
    names = ["train_state.yaml", "model.pt", "train_config.yaml", "train_checkpoint.pt"]

    corrupted: List[Tuple[str, str]] = []  # name: cause
    run_info: List[Dict[str, Any]] = []
    num_models = 0
    for path in sweep_dir.iterdir():
        if not path.is_dir():
            continue
        if all(not (path / name).exists() for name in names):  # not a model folder
            continue
        num_models += 1

        # If there are some files but no train config, something's wrong.
        if not (path / "train_config.yaml").exists():
            missing_files = [name for name in names if not (path / name).exists()]
            corrupted.append(
                (
                    path.name,
                    f"Missing train_config.yaml. Missing files: {missing_files}",
                )
            )
            continue

        # If no train state, zero progress.
        if not (path / "train_state.yaml").exists():
            # Read max step from config.
            with open((path / "train_config.yaml")) as f:
                train_config = yaml.safe_load(f)
            run_info.append(
                {
                    "name": path.name,
                    "step": 0,
                    "max_step": train_config["trainer"]["steps"],
                    "progress": 0.0,
                }
            )
            continue

        with open((path / "train_state.yaml")) as f:
            train_state = yaml.safe_load(f)
        step = train_state["step"]
        max_step = train_state["max_step"]
        if step < 1 or step > max_step or max_step < 1:
            corrupted.append((path.name, f"step: {step}, max_step: {max_step}"))
            continue

        # Save run info for non-corrupted jobs.
        run_info.append(
            {
                "name": path.name,
                **train_state,
                "progress": train_state["step"] / train_state["max_step"],
            }
        )

    df = pd.DataFrame(run_info)
    overall_progress = df["step"].sum() / df["max_step"].sum()
    num_finished = (df["step"] == df["max_step"]).sum()
    perc_finished = num_finished / len(df)

    print(f"Num models: {num_models}")
    if len(corrupted) > 0:
        print(f"Corrupted: {len(corrupted)} ({len(corrupted) / num_models * 100:.1f}%)")
        print("List corrupted:")
        for name, reason in corrupted:
            print(f"    {name}: {reason}")
    extra = " of non-corrupted" if len(corrupted) > 0 else ""
    print(f"Num finished: {num_finished} ({perc_finished * 100:.1f}%{extra})")
    extra = " (of non-corrupted)" if len(corrupted) > 0 else ""
    print(f"Overall progress{extra}: {overall_progress * 100:.1f}%")
    extra = " and non-corrupted" if len(corrupted) > 0 else ""

    # Unfinished and not corrupted
    df_running = df[df["step"] < df["max_step"]].copy()

    if len(df_running) > 0:
        progress_running = df_running["step"].sum() / df_running["max_step"].sum()
        print(f"Progress of unfinished{extra} runs: {progress_running * 100:.1f}%")
        print("Details:")
        # Sort by progress, then show info on each unfinished job.
        df_running.sort_values("progress", ascending=False, inplace=True)
        for row in df_running.itertuples():
            fill_len = int(np.round(bar_len * row.progress))
            bar = FILL_CHAR * fill_len + EMPTY_CHAR * (bar_len - fill_len)
            # TODO make formatting more general: use max len of run name, and k/M/... for steps.
            print(
                f"{row.name:6s} {bar} {row.progress * 100:.1f}% "
                f"({row.step // 1000}k/{row.max_step // 1000}k)"
            )
