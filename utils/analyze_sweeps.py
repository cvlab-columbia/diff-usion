from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout
import pandas as pd
import numpy as np


def analyze_results_like_baseline(
    df: pd.DataFrame, log_path=None, manip_values: Optional[list[float]] = None
):
    # Extract parameters from experiment names
    df["manip"] = df["experiment"].apply(lambda x: float(x.split("_")[3]))
    df["tksip"] = df["experiment"].apply(lambda x: float(x.split("_")[1]))
    df["gs_tar"] = df["experiment"].apply(lambda x: int(x.split("_")[5]))

    # Get unique values for each parameter
    if manip_values is None:
        manip_values = sorted(df["manip"].unique(), reverse=False)
    tksip_values = sorted(df["tksip"].unique(), reverse=True)  # Now in descending order
    gs_values = sorted(df["gs_tar"].unique())

    print(f"Searching through:")
    print(f"Manipulation values: {manip_values}")
    print(f"TKSIP values (descending): {tksip_values}")
    print(f"Guidance scale values: {gs_values}")

    # For each image
    results = {}
    total_images = len(df["filename"].unique())
    successful_flips = 0

    for filename in df["filename"].unique():
        img_df = df[df["filename"] == filename]
        target = img_df.iloc[0]["target"]

        # Search through parameters in order (like the baseline)
        found = False
        for m in manip_values:
            for t in tksip_values:  # Now trying higher tksip values first
                for g in gs_values:
                    # Get results for this parameter combination
                    exp_results = img_df[
                        (img_df["manip"] == m)
                        & (img_df["tksip"] == t)
                        & (img_df["gs_tar"] == g)
                    ]

                    if len(exp_results) == 0:
                        continue

                    # Check if this flips the label (pred is opposite of target)
                    if (target == 0 and exp_results.iloc[0]["pred"] == 1) or (
                        target == 1 and exp_results.iloc[0]["pred"] == 0
                    ):
                        results[filename] = {
                            "manip": m,
                            "tksip": t,
                            "gs_tar": g,
                            "lpips": exp_results.iloc[0]["lpips"],
                            "pred": exp_results.iloc[0]["pred"],
                            "target": target,
                        }
                        successful_flips += 1
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            results[filename] = {
                "manip": m,
                "tksip": t,
                "gs_tar": g,
                "lpips": exp_results.iloc[0]["lpips"],
                "pred": exp_results.iloc[0]["pred"],
                "target": target,
            }

    results_df = pd.DataFrame.from_dict(results, orient="index")

    flip_rate = successful_flips / total_images

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nSummary:")
                print(f"Total images: {total_images}")
                print(f"Successfully flipped: {successful_flips}")
                print(f"Flip rate: {flip_rate:.2%}")
                print("\nParameter distribution in successful flips:")
                print("\nManipulation strength:")
                print(results_df["manip"].value_counts().sort_index())
                print("\nTKSIP:")
                print(results_df["tksip"].value_counts().sort_index())
                print("\nGuidance scale:")
                print(results_df["gs_tar"].value_counts().sort_index())
                print(f"\nAverage LPIPS: {results_df['lpips'].mean():.4f}")

    return results_df, flip_rate


def analyze_ddpmef_results(df: pd.DataFrame, log_path=None):
    # Extract parameters from experiment names like 'skip_0.4_manip_1_cfgtar_4_mode_ManipulateMode.cond_avg'
    def parse_experiment(exp):
        parts = exp.split("_")
        skip = round(float(parts[1]), 2)
        cfgtar = int(parts[-1])
        return pd.Series({"skip": skip, "cfgtar": cfgtar})

    # Apply parsing to create new columns
    df[["skip", "cfgtar"]] = df["experiment"].apply(parse_experiment)

    # Get unique values for each parameter
    skip_values = sorted(df["skip"].unique(), reverse=True)  # descending order
    cfgtar_values = sorted(df["cfgtar"].unique())

    print(f"Searching through:")
    print(f"Skip values (descending): {skip_values}")
    print(f"CFG target values: {cfgtar_values}")

    # For each image
    results = {}
    total_images = len(df["filename"].unique())
    successful_flips = 0

    for filename in df["filename"].unique():
        img_df = df[df["filename"] == filename]
        target = img_df.iloc[0]["target"]

        # Search through parameters in order
        found = False
        for skip in skip_values:  # trying higher skip values first
            for cfg in cfgtar_values:
                # Get results for this parameter combination
                exp_results = img_df[
                    (img_df["skip"] == skip) & (img_df["cfgtar"] == cfg)
                ]

                if len(exp_results) == 0:
                    continue

                # Check if this flips the label (pred is opposite of target)
                if (target == 0 and exp_results.iloc[0]["pred"] == 1) or (
                    target == 1 and exp_results.iloc[0]["pred"] == 0
                ):
                    # if (target == 0 and exp_results.iloc[0]["pred"] == 0) or (
                    #     target == 1 and exp_results.iloc[0]["pred"] == 1
                    # ):
                    results[filename] = {
                        "skip": skip,
                        "cfgtar": cfg,
                        "lpips": exp_results.iloc[0]["lpips"],
                        "pred": exp_results.iloc[0]["pred"],
                        "target": target,
                    }
                    successful_flips += 1
                    found = True
                    break
            if found:
                break

        if not found:
            results[filename] = {
                "skip": skip,
                "cfgtar": cfg,
                "lpips": exp_results.iloc[0]["lpips"],
                "pred": exp_results.iloc[0]["pred"],
                "target": target,
            }

    results_df = pd.DataFrame.from_dict(results, orient="index")

    flip_rate = successful_flips / total_images

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nSummary:")
                print(f"Total images: {total_images}")
                print(f"Successfully flipped: {successful_flips}")
                print(f"Flip rate: {flip_rate:.2%}")
                print("\nParameter distribution in successful flips:")
                print("\nSkip values:")
                print(results_df["skip"].value_counts().sort_index())
                print("\nCFG target values:")
                print(results_df["cfgtar"].value_counts().sort_index())
                print(f"\nAverage LPIPS: {results_df['lpips'].mean():.4f}")

    return results_df, flip_rate


def analyze_visual_sliders_results(df: pd.DataFrame, log_path=None):
    # Extract parameters from experiment names like 'config_a1.0_r4_n2_dataset_10K_alpha1.0_rank4_noxattn_last.pt'
    def parse_experiment(exp):
        # Remove .pt extension and split
        parts = exp.replace(".pt", "").split("_")
        rank = int(parts[2][1:])  # r4 -> 4
        n = int(parts[3][1:])  # n2 -> 2
        return pd.Series({"rank": rank, "n": n})

    df[["rank", "n"]] = df["experiment"].apply(parse_experiment)

    params = {"rank": sorted(df["rank"].unique()), "n": sorted(df["n"].unique())}

    print(f"Parameter ranges:")
    for param, values in params.items():
        print(f"{param}: {values}")

    # For each image
    results = {}
    total_images = len(df["filename"].unique())
    successful_flips = 0

    for filename in df["filename"].unique():
        img_df = df[df["filename"] == filename]
        target = img_df.iloc[0]["target"]

        # Search through parameters in order
        found = False
        for rank in params["rank"]:
            for n in params["n"]:
                exp_results = img_df[(img_df["rank"] == rank) & (img_df["n"] == n)]

                if len(exp_results) == 0:
                    continue

                if (target == 0 and exp_results.iloc[0]["pred"] == 1) or (
                    target == 1 and exp_results.iloc[0]["pred"] == 0
                ):
                    results[filename] = {
                        "rank": rank,
                        "n": n,
                        "experiment": exp_results.iloc[0]["experiment"],
                        "lpips": exp_results.iloc[0]["lpips"],
                        "pred": exp_results.iloc[0]["pred"],
                        "target": target,
                    }
                    successful_flips += 1
                    found = True
                    break
            if found:
                break

        if not found:
            results[filename] = {
                "rank": rank,
                "n": n,
                "experiment": exp_results.iloc[0]["experiment"],
                "lpips": exp_results.iloc[0]["lpips"],
                "pred": exp_results.iloc[0]["pred"],
                "target": target,
            }
    results_df = pd.DataFrame.from_dict(results, orient="index")

    flip_rate = successful_flips / total_images

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nSummary:")
                print(f"Total images: {total_images}")
                print(f"Successfully flipped: {successful_flips}")
                print(f"Flip rate: {flip_rate:.2%}")
                print("\nParameter distribution in successful flips:")
                print("\nRank values:")
                print(results_df["rank"].value_counts().sort_index())
                print("\nN values:")
                print(results_df["n"].value_counts().sort_index())
                print(f"\nAverage LPIPS: {results_df['lpips'].mean():.4f}")

    return results_df, flip_rate


def main():
    # Read and analyze
    sweep_path = Path(
        "/proj/vondrick2/orr/projects/magnification/results/eval/kandinsky_sweeps/reports_orig_embeds/afhq/report.csv"
    )
    log_path = sweep_path.parent / "log_unflip.txt"

    # df = pd.read_csv('results_logs/method_comparison_results_kikibouba_visual_sliders.csv')
    # results, flip_rate = analyze_visual_sliders_results(df)

    # DDPMEF
    # df = pd.read_csv(sweep_path)
    # results, flip_rate = analyze_ddpmef_results(df, log_path)
    # results.to_csv(sweep_path.parent / f"analyzed-unflip-{sweep_path.name}")

    # Ours
    df = pd.read_csv(sweep_path)
    results, flip_rate = analyze_results_like_baseline(df, log_path, manip_values=[1])
    results.to_csv(sweep_path.parent / f"analyzed-unflip-{sweep_path.name}")

    # Sliders
    # df = pd.read_csv(sweep_path)
    # results, flip_rate = analyze_visual_sliders_results(df)
    # results.to_csv(sweep_path.parent / f"analyzed-unflip-{sweep_path.name}")


if __name__ == "__main__":
    main()
