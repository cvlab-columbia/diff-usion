from pathlib import Path
from contextlib import redirect_stdout
import pandas as pd


def analyze_results_like_baseline(df: pd.DataFrame, log_path=None):
    # Extract parameters from experiment names
    df["manip"] = df["experiment"].apply(lambda x: float(x.split("_")[3]))
    df["tksip"] = df["experiment"].apply(lambda x: float(x.split("_")[1]))
    df["gs_tar"] = df["experiment"].apply(lambda x: int(x.split("_")[5]))

    # Get unique values for each parameter
    manip_values = sorted(df["manip"].unique(), reverse=False)
    tksip_values = sorted(df["tksip"].unique(), reverse=True)  # Now in descending order
    gs_values = sorted(df["gs_tar"].unique())

    print(f"Searching through:")
    print(f"Manipulation values: {manip_values}")
    print(f"TKSIP values (descending): {tksip_values}")
    print(f"Guidance scale values: {gs_values}")

    # Store results for each manipulation value
    results_by_manip = {m: {} for m in manip_values}
    total_images = len(df["filename"].unique())
    successful_flips_by_manip = {m: 0 for m in manip_values}

    for filename in df["filename"].unique():
        img_df = df[df["filename"] == filename]
        target = img_df.iloc[0]["target"]

        # Search through parameters for each manipulation value separately
        for m in manip_values:
            found = False
            for t in tksip_values:
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
                        results_by_manip[m][filename] = {
                            "manip": m,
                            "tksip": t,
                            "gs_tar": g,
                            "lpips": exp_results.iloc[0]["lpips"],
                            "pred": exp_results.iloc[0]["pred"],
                            "target": target,
                        }
                        successful_flips_by_manip[m] += 1
                        found = True
                        break
                if found:
                    break

    # Create DataFrames for each manipulation value
    results_dfs = {}
    flip_rates = {}

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nResults by manipulation value:")
                for m in manip_values:
                    results_df = pd.DataFrame.from_dict(
                        results_by_manip[m], orient="index"
                    )
                    results_dfs[m] = results_df
                    flip_rates[m] = successful_flips_by_manip[m] / total_images

                    print(f"\n=== Manipulation value: {m} ===")
                    print(f"Total images: {total_images}")
                    print(f"Successfully flipped: {successful_flips_by_manip[m]}")
                    print(f"Flip rate: {flip_rates[m]:.2%}")

                    if not results_df.empty:
                        print("\nParameter distribution in successful flips:")
                        print("\nTKSIP:")
                        print(results_df["tksip"].value_counts().sort_index())
                        print("\nGuidance scale:")
                        print(results_df["gs_tar"].value_counts().sort_index())
                        print(f"\nAverage LPIPS: {results_df['lpips'].mean():.4f}")
                    else:
                        print("\nNo successful flips for this manipulation value")

    return results_dfs, flip_rates


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


def analyze_visual_sliders_results(df, log_path=None):
    # Extract parameters from experiment names like 'config_a1.0_r4_n2_dataset_10K_alpha1.0_rank4_noxattn_last.pt'
    def parse_experiment(exp):
        # Remove .pt extension and split
        parts = exp.replace(".pt", "").split("_")
        rank = int(parts[2][1:])  # r4 -> 4
        n = int(parts[3][1:])  # n2 -> 2
        return pd.Series({"rank": rank, "n": n})

    df[["rank", "n"]] = df["experiment"].apply(parse_experiment)

    params = {"rank": sorted(df["rank"].unique()), "n": sorted(df["n"].unique())}
    total_images = len(df["filename"].unique())

    print(f"Parameter ranges:")
    for param, values in params.items():
        print(f"{param}: {values}")

    # Try each combination of parameters
    best_combo = None
    best_flip_rate = 0
    best_avg_lpips = float("inf")
    best_results = None
    params["n"] = [5, 10, 20, 1000]

    for rank in params["rank"]:
        for n in params["n"]:
            # Get results for this parameter combination
            results = {}
            successful_flips = 0

            for filename in df["filename"].unique():
                img_df = df[df["filename"] == filename]
                target = img_df.iloc[0]["target"]

                exp_results = img_df[(img_df["rank"] == rank) & (img_df["n"] == n)]

                if len(exp_results) > 0 and (
                    (target == 1 and exp_results.iloc[0]["pred"] == 1)
                    or (target == 0 and exp_results.iloc[0]["pred"] == 0)
                ):
                    results[filename] = {
                        "rank": rank,
                        "n": n,
                        "lpips": exp_results.iloc[0]["lpips"],
                        "pred": exp_results.iloc[0]["pred"],
                        "target": target,
                    }
                    successful_flips += 1

            if successful_flips > 0:
                results_df = pd.DataFrame.from_dict(results, orient="index")
                flip_rate = successful_flips / total_images
                avg_lpips = results_df["lpips"].mean()

                # Update best combination if:
                # 1. Higher flip rate, or
                # 2. Same flip rate but lower LPIPS
                if flip_rate > best_flip_rate or (
                    flip_rate == best_flip_rate and avg_lpips < best_avg_lpips
                ):
                    best_combo = (rank, n)
                    best_flip_rate = flip_rate
                    best_avg_lpips = avg_lpips
                    best_results = results_df

    if best_combo is None:
        print("No successful parameter combinations found")
        return None, 0

    print("\nBest parameter combination:")
    print(f"Rank: {best_combo[0]}, n: {best_combo[1]}")
    print("\nSummary:")
    print(f"Total images: {total_images}")
    print(f"Successfully flipped: {len(best_results)}")
    print(f"Flip rate: {best_flip_rate:.2%}")
    print(f"Average LPIPS: {best_avg_lpips:.4f}")

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nBest parameter combination:")
                print(f"Rank: {best_combo[0]}, n: {best_combo[1]}")
                print("\nSummary:")
                print(f"Total images: {total_images}")
                print(f"Successfully flipped: {len(best_results)}")
                print(f"Flip rate: {best_flip_rate:.2%}")
                print(f"Average LPIPS: {best_avg_lpips:.4f}")

    return best_results, best_flip_rate


def analyze_results_from_best(df: pd.DataFrame, samples_dir: Path, log_path=None):
    """
    Analyze results using the BEST_ prefixed files as the best results.

    Args:
        df: DataFrame containing the report data
        samples_dir: Directory containing the sample images with BEST_ prefix
        log_path: Path to save the analysis log

    Returns:
        Tuple of (results_dfs, flip_rates) where:
            - results_dfs is a dict mapping manipulation values to DataFrames of results
            - flip_rates is a dict mapping manipulation values to flip rates
    """
    # Extract parameters from experiment names
    # import pdb; pdb.set_trace()
    # filter df for rows where experiment contains BEST_
    df = df[df["experiment"].str.contains("BEST_")]
    # Extract parameters from experiment names
    df["manip"] = df["experiment"].apply(
        lambda x: float(x.split("_")[4]) if "manip_" in x else 0.0
    )

    # Extract parameters from experiment names
    df["tksip"] = df["experiment"].apply(lambda x: float(x.split("_")[2]))
    df["gs_tar"] = df["experiment"].apply(lambda x: int(x.split("_")[6]))

    # Get unique values for each parameter
    manip_values = sorted(df["manip"].unique(), reverse=False)
    tksip_values = sorted(df["tksip"].unique(), reverse=True)  # Now in descending order
    gs_values = sorted(df["gs_tar"].unique())

    print(f"Searching through:")
    print(f"Manipulation values: {manip_values}")
    print(f"TKSIP values (descending): {tksip_values}")
    print(f"Guidance scale values: {gs_values}")

    # Store results for each manipulation value
    results_by_manip = {m: {} for m in manip_values}
    total_images = len(df["filename"].unique())
    successful_flips_by_manip = {m: 0 for m in manip_values}

    for filename in df["filename"].unique():
        img_df = df[df["filename"] == filename]
        target = img_df.iloc[0]["target"]

        # Search through parameters for each manipulation value separately
        for m in manip_values:
            found = False
            for t in tksip_values:
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
                        results_by_manip[m][filename] = {
                            "manip": m,
                            "tksip": t,
                            "gs_tar": g,
                            "lpips": exp_results.iloc[0]["lpips"],
                            "pred": exp_results.iloc[0]["pred"],
                            "target": target,
                        }
                        successful_flips_by_manip[m] += 1
                        found = True
                        break
                if found:
                    break

    # Create DataFrames for each manipulation value
    results_dfs = {}
    flip_rates = {}

    if log_path is not None:
        with open(log_path, "w") as f:
            with redirect_stdout(f):
                print("\nResults by manipulation value:")
                for m in manip_values:
                    results_df = pd.DataFrame.from_dict(
                        results_by_manip[m], orient="index"
                    )
                    results_dfs[m] = results_df
                    flip_rates[m] = successful_flips_by_manip[m] / total_images

                    print(f"\n=== Manipulation value: {m} ===")
                    print(f"Total images: {total_images}")
                    print(f"Successfully flipped: {successful_flips_by_manip[m]}")
                    print(f"Flip rate: {flip_rates[m]:.2%}")

                    if not results_df.empty:
                        print("\nParameter distribution in successful flips:")
                        print("\nTKSIP:")
                        print(results_df["tksip"].value_counts().sort_index())
                        print("\nGuidance scale:")
                        print(results_df["gs_tar"].value_counts().sort_index())
                        print(f"\nAverage LPIPS: {results_df['lpips'].mean():.4f}")
                    else:
                        print("\nNo successful flips for this manipulation value")

    return results_dfs, flip_rates


def main():
    # Read and analyze
    sweep_path = Path("./reports/report.csv")
    log_path = sweep_path.parent / "log_manip2.txt"

    # Read and analyze
    df = pd.read_csv(sweep_path)
    results, flip_rate = analyze_results_from_best(df, sweep_path.parent, log_path)
    results.to_csv(sweep_path.parent / f"analyzed-manip2-{sweep_path.name}")


if __name__ == "__main__":
    main()
