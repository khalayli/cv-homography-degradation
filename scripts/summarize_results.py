import argparse
import math
import os
import sys

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))

if REPO_ROOT not in sys.path:
    print(f"[bootstrap] Adding repo root to sys.path: {REPO_ROOT}")
    sys.path.insert(0, REPO_ROOT)

from src.utils.io import ensure_dir, write_json


def parse_args():
    print("[parse_args] Parsing command line arguments for summarize_results.py")
    parser = argparse.ArgumentParser(description="Aggregate metrics.csv into a smaller summary.")
    parser.add_argument("--input", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override output directory")
    return parser.parse_args()


def read_rows(path):
    print(f"[read_rows] Reading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"[read_rows] DataFrame shape={df.shape}")
    print(f"[read_rows] Columns={list(df.columns)}")
    return df


def _make_numeric(df, columns):
    print(f"[_make_numeric] Converting columns to numeric: {columns}")
    out = df.copy()

    for col in columns:
        if col in out.columns:
            print(f"[_make_numeric] Converting column: {col}")
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            print(f"[_make_numeric] Skipping missing column: {col}")

    return out


def aggregate(df):
    print(f"[aggregate] Starting aggregation, input shape={df.shape}")

    required_group_cols = ["matcher", "corruption", "severity"]
    for col in required_group_cols:
        if col not in df.columns:
            raise ValueError(f"[aggregate] Missing required column: {col}")

    numeric_cols = ["mean_corner_error", "runtime_s", "homography_success"]
    success_cols = [col for col in df.columns if col.startswith("success@")]

    print(f"[aggregate] success_cols={success_cols}")

    df = _make_numeric(df, numeric_cols + success_cols + ["severity"])

    agg_dict = {
        "mean_corner_error": "mean",
        "runtime_s": "mean",
        "homography_success": "mean",
    }

    for col in success_cols:
        agg_dict[col] = "mean"

    print(f"[aggregate] agg_dict={agg_dict}")

    grouped = (
        df.groupby(["matcher", "corruption", "severity"], dropna=False)
        .agg(
            num_pairs=("matcher", "size"),
            **{col: (col, agg_fn) for col, agg_fn in agg_dict.items()}
        )
        .reset_index()
    )

    print(f"[aggregate] Grouped shape before rename={grouped.shape}")

    grouped = grouped.rename(
        columns={
            "runtime_s": "mean_runtime_s",
            "homography_success": "homography_return_rate",
        }
    )

    if "success@3" in grouped.columns:
        print("[aggregate] Creating success_rate_3px from success@3")
        grouped["success_rate_3px"] = grouped["success@3"]

    if "success@5" in grouped.columns:
        print("[aggregate] Creating success_rate_5px from success@5")
        grouped["success_rate_5px"] = grouped["success@5"]

    if "mean_corner_error" in grouped.columns:
        print("[aggregate] Filling NaN mean_corner_error with inf")
        grouped["mean_corner_error"] = grouped["mean_corner_error"].fillna(float("inf"))

    if "mean_runtime_s" in grouped.columns:
        print("[aggregate] Leaving NaN mean_runtime_s as-is")

    if "homography_return_rate" in grouped.columns:
        print("[aggregate] Filling NaN homography_return_rate with 0.0")
        grouped["homography_return_rate"] = grouped["homography_return_rate"].fillna(0.0)

    for col in success_cols:
        if col in grouped.columns:
            print(f"[aggregate] Filling NaN in {col} with 0.0")
            grouped[col] = grouped[col].fillna(0.0)

    if "success_rate_3px" in grouped.columns:
        print("[aggregate] Filling NaN success_rate_3px with 0.0")
        grouped["success_rate_3px"] = grouped["success_rate_3px"].fillna(0.0)

    if "success_rate_5px" in grouped.columns:
        print("[aggregate] Filling NaN success_rate_5px with 0.0")
        grouped["success_rate_5px"] = grouped["success_rate_5px"].fillna(0.0)

    grouped = grouped.sort_values(
        by=["matcher", "corruption", "severity"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    print(f"[aggregate] Final grouped shape={grouped.shape}")
    print("[aggregate] Preview:")
    print(grouped.head())

    return grouped


def write_csv(path, df):
    print(f"[write_csv] Writing summary CSV to: {path}")
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)
    print("[write_csv] Done writing summary CSV")


def dataframe_to_json_rows(df):
    print("[dataframe_to_json_rows] Converting DataFrame to JSON-safe rows")
    rows = df.to_dict(orient="records")

    for row_idx, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                print(f"[dataframe_to_json_rows] row_idx={row_idx}, key={key}, replacing NaN with None")
                row[key] = None

    return rows


def main():
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir or os.path.dirname(input_path) or "."

    print(f"[main] input_path={input_path}")
    print(f"[main] output_dir={output_dir}")

    df = read_rows(input_path)
    summary_df = aggregate(df)

    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary_aggregated.json")

    write_csv(csv_path, summary_df)

    json_rows = dataframe_to_json_rows(summary_df)
    write_json(json_path, {"rows": json_rows})

    print(f"[main] Wrote csv_path={csv_path}")
    print(f"[main] Wrote json_path={json_path}")


if __name__ == "__main__":
    main()
