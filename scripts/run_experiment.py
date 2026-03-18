import argparse
import csv
import os
import sys

CURRENT_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))

if REPO_ROOT not in sys.path:
    print(f"[bootstrap] Adding repo root to sys.path: {REPO_ROOT}")
    sys.path.insert(0, REPO_ROOT)

from src.data.corruptions import apply_corruption, apply_corruption_sequence
from src.data.hpatches import HPatchesDataset
from src.geom.homography import estimate_homography
from src.geom.metrics import build_pair_metrics
from src.matchers import build_matcher
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.seeding import seed_everything
from src.utils.timing import timer


def parse_args():
    print("[parse_args] Parsing command line arguments for run_experiment.py")
    parser = argparse.ArgumentParser(description="Run homography experiments on HPatches.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Optional small limit for quick smoke tests.",
    )
    return parser.parse_args()


def build_corruption_settings(corruption_cfg):
    print(f"[build_corruption_settings] corruption_cfg={corruption_cfg}")

    enabled = bool(corruption_cfg.get("enabled", False))
    include_clean = bool(corruption_cfg.get("include_clean", True))
    mode = corruption_cfg.get("mode", "single")
    types = list(corruption_cfg.get("types", []))
    severities = list(corruption_cfg.get("severities", [0]))

    settings = []

    if include_clean:
        settings.append({
            "name": "none",
            "severity": 0,
            "kinds": [],
        })

    if not enabled:
        print(f"[build_corruption_settings] Corruptions disabled, settings={settings}")
        return settings

    positive_severities = []
    for s in severities:
        s = int(s)
        if s > 0:
            positive_severities.append(s)

    if mode == "single":
        for kind in types:
            for severity in positive_severities:
                settings.append({
                    "name": kind,
                    "severity": severity,
                    "kinds": [kind],
                })
        print(f"[build_corruption_settings] Built single-corruption settings: {settings}")
        return settings

    if mode == "sequence":
        for severity in positive_severities:
            settings.append({
                "name": "sequence",
                "severity": severity,
                "kinds": types,
            })
        print(f"[build_corruption_settings] Built sequence-corruption settings: {settings}")
        return settings

    raise ValueError(f"Unknown corruption mode: {mode}")


def apply_setting(image0, image1, setting, apply_to):
    print(f"[apply_setting] setting={setting}, apply_to={apply_to}")

    if setting["name"] == "none" or setting["severity"] == 0 or not setting["kinds"]:
        print("[apply_setting] Clean setting selected, returning original images")
        return image0, image1

    out0 = image0.copy()
    out1 = image1.copy()

    if len(setting["kinds"]) == 1:
        kind = setting["kinds"][0]

        if apply_to in ["source", "both"]:
            out0 = apply_corruption(out0, kind=kind, severity=setting["severity"])

        if apply_to in ["target", "both"]:
            out1 = apply_corruption(out1, kind=kind, severity=setting["severity"])

        print("[apply_setting] Applied single corruption")
        return out0, out1

    if apply_to in ["source", "both"]:
        out0 = apply_corruption_sequence(
            out0,
            corruption_types=setting["kinds"],
            severity=setting["severity"],
        )

    if apply_to in ["target", "both"]:
        out1 = apply_corruption_sequence(
            out1,
            corruption_types=setting["kinds"],
            severity=setting["severity"],
        )

    print("[apply_setting] Applied corruption sequence")
    return out0, out1


def write_csv(path, rows):
    print(f"[write_csv] path={path}, num_rows={len(rows)}")
    ensure_dir(os.path.dirname(path) or ".")

    if not rows:
        print("[write_csv] No rows to write, creating empty file with no header")
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[write_csv] Done writing CSV")


def get_value(obj, key, default=None):
    if obj is None:
        print(f"[get_value] obj is None for key='{key}', returning default={default}")
        return default

    if isinstance(obj, dict):
        value = obj.get(key, default)
        print(f"[get_value] Got dict key='{key}', value type={type(value)}")
        return value

    if hasattr(obj, key):
        value = getattr(obj, key)
        print(f"[get_value] Got attribute='{key}', value type={type(value)}")
        return value

    print(f"[get_value] Missing key='{key}', returning default={default}")
    return default


def get_dataset_items(dataset):
    if hasattr(dataset, "pairs"):
        print("[get_dataset_items] Using dataset.pairs")
        return dataset.pairs

    if hasattr(dataset, "samples"):
        print("[get_dataset_items] Using dataset.samples")
        return dataset.samples

    print("[get_dataset_items] Dataset has no .pairs or .samples")
    raise AttributeError("Dataset must expose either .pairs or .samples")


def set_dataset_items(dataset, new_items):
    if hasattr(dataset, "pairs"):
        print("[set_dataset_items] Writing into dataset.pairs")
        dataset.pairs = new_items
        return

    if hasattr(dataset, "samples"):
        print("[set_dataset_items] Writing into dataset.samples")
        dataset.samples = new_items
        return

    print("[set_dataset_items] Dataset has no .pairs or .samples")
    raise AttributeError("Dataset must expose either .pairs or .samples")


def extract_images(example):
    print("[extract_images] Extracting images from dataset example")

    image0 = get_value(example, "image0")
    image1 = get_value(example, "image1")

    if image0 is None:
        image0 = get_value(example, "image_src")
    if image1 is None:
        image1 = get_value(example, "image_tgt")

    if image0 is None or image1 is None:
        raise ValueError("Dataset example must contain image0/image1 or image_src/image_tgt")

    return image0, image1


def extract_metadata(example):
    print("[extract_metadata] Extracting metadata from dataset example")

    pair_obj = get_value(example, "pair")

    scene_name = get_value(example, "scene_name")
    if scene_name is None:
        scene_name = get_value(example, "scene")
    if scene_name is None and pair_obj is not None:
        scene_name = get_value(pair_obj, "scene_name")

    split = get_value(example, "split")
    if split is None and pair_obj is not None:
        split = get_value(pair_obj, "split")

    pair_index = get_value(example, "pair_index")
    if pair_index is None:
        pair_index = get_value(example, "pair")
    if pair_index is None and pair_obj is not None:
        pair_index = get_value(pair_obj, "pair_index")

    H_gt = get_value(example, "homography_gt")
    if H_gt is None:
        H_gt = get_value(example, "H_gt")

    return {
        "scene_name": scene_name,
        "split": split,
        "pair_index": pair_index,
        "H_gt": H_gt,
    }


def extract_matched_points(match_result):
    print("[extract_matched_points] Extracting matched points from matcher output")

    matched_points0 = get_value(match_result, "matched_points0")
    matched_points1 = get_value(match_result, "matched_points1")

    if matched_points0 is None or matched_points1 is None:
        raise ValueError("match_result must contain matched_points0 and matched_points1")

    return matched_points0, matched_points1


def get_optional_geom_overrides(match_result):
    print("[get_optional_geom_overrides] Checking matcher output for geometry overrides...")

    geom_overrides = get_value(match_result, "geom_overrides", {})

    if geom_overrides is None:
        print("[get_optional_geom_overrides] geom_overrides is None, using {}")
        return {}

    if not isinstance(geom_overrides, dict):
        print(f"[get_optional_geom_overrides] Invalid geom_overrides type={type(geom_overrides)}, using {{}}")
        return {}

    print(f"[get_optional_geom_overrides] geom_overrides={geom_overrides}")
    return geom_overrides


def resolve_geom_settings(cfg, geom_overrides):
    print("[resolve_geom_settings] Resolving geometry settings...")
    print(f"[resolve_geom_settings] geom_overrides={geom_overrides}")

    default_geom = cfg.get("geom", {})

    reproj_threshold = float(
        geom_overrides.get(
            "reproj_threshold",
            default_geom.get("reproj_threshold", 3.0),
        )
    )

    confidence = float(
        geom_overrides.get(
            "confidence",
            default_geom.get("confidence", 0.999),
        )
    )

    max_iters = int(
        geom_overrides.get(
            "max_iters",
            default_geom.get("max_iters", 5000),
        )
    )

    resolved = {
        "reproj_threshold": reproj_threshold,
        "confidence": confidence,
        "max_iters": max_iters,
    }

    print(f"[resolve_geom_settings] resolved={resolved}")
    return resolved


def add_matcher_debug_fields(row, match_result, geom_settings):
    print("[add_matcher_debug_fields] Adding matcher/debug fields to row")

    row["backend_used"] = get_value(match_result, "backend_used", "")
    row["fallback_used"] = int(get_value(match_result, "fallback_used", 0) or 0)
    row["used_reproj_threshold"] = float(geom_settings["reproj_threshold"])
    row["used_confidence"] = float(geom_settings["confidence"])
    row["used_max_iters"] = int(geom_settings["max_iters"])

    print(f"[add_matcher_debug_fields] row backend_used={row['backend_used']}")
    print(f"[add_matcher_debug_fields] row fallback_used={row['fallback_used']}")
    print(f"[add_matcher_debug_fields] row used_reproj_threshold={row['used_reproj_threshold']}")


def main():
    args = parse_args()
    cfg = read_yaml(args.config)

    out_dir = cfg["run"]["out_dir"]
    ensure_dir(out_dir)

    seed_everything(int(cfg["run"]["seed"]))

    print("[main] Building dataset...")
    dataset = HPatchesDataset(
        root=cfg["dataset"]["hpatches_root"],
        pairs_per_scene=int(cfg["dataset"].get("pairs_per_scene", 5)),
    )

    print("[main] Building matcher...")
    matcher = build_matcher(cfg["method"])

    print("[main] Building corruption settings...")
    corruption_settings = build_corruption_settings(cfg.get("corruptions", {}))

    thresholds = cfg["eval"].get("corner_error_thresholds", [1, 3, 5, 10])
    apply_to = cfg.get("corruptions", {}).get("apply_to", "target")

    dataset_items = get_dataset_items(dataset)

    if args.limit_pairs is not None:
        print(f"[main] Applying limit_pairs={args.limit_pairs}")
        dataset_items = dataset_items[:args.limit_pairs]
        set_dataset_items(dataset, dataset_items)

    rows = []

    print(f"[main] Number of dataset pairs to process: {len(dataset_items)}")

    for setting in corruption_settings:
        print(f"[main] Running setting: {setting}")

        for pair_idx in range(len(dataset_items)):
            print(f"[main] Processing pair_idx={pair_idx}")

            example = dataset[pair_idx]
            image0, image1 = extract_images(example)
            meta = extract_metadata(example)

            eval_image0, eval_image1 = apply_setting(
                image0,
                image1,
                setting=setting,
                apply_to=apply_to,
            )

            with timer(
                f"pair::{meta['scene_name']}::{setting['name']}::s{setting['severity']}"
            ) as t:
                match_result = matcher.match(eval_image0, eval_image1)
                matched_points0, matched_points1 = extract_matched_points(match_result)

                geom_overrides = get_optional_geom_overrides(match_result)
                geom_settings = resolve_geom_settings(cfg, geom_overrides)

                H_result = estimate_homography(
                    matched_points0,
                    matched_points1,
                    reproj_threshold=geom_settings["reproj_threshold"],
                    confidence=geom_settings["confidence"],
                    max_iters=geom_settings["max_iters"],
                )

            image_height, image_width = eval_image0.shape[:2]

            row = build_pair_metrics(
                scene_name=meta["scene_name"],
                split=meta["split"],
                pair_index=meta["pair_index"],
                matcher_name=matcher.name,
                corruption_name=setting["name"],
                severity=setting["severity"],
                runtime_s=float(t.elapsed_s or 0.0),
                image_width=int(image_width),
                image_height=int(image_height),
                H_gt=meta["H_gt"],
                H_result=H_result,
                thresholds=thresholds,
            )

            add_matcher_debug_fields(row, match_result, geom_settings)

            rows.append(row)
            print(f"[main] Added row with mean_corner_error={row.get('mean_corner_error')}")

    metrics_csv_path = os.path.join(out_dir, "metrics.csv")
    summary_json_path = os.path.join(out_dir, "summary.json")
    settings_json_path = os.path.join(out_dir, "resolved_settings.json")

    write_csv(metrics_csv_path, rows)

    summary = {
        "run_name": cfg["run"]["name"],
        "matcher": matcher.name,
        "num_rows": len(rows),
        "corruption_settings": corruption_settings,
    }

    if hasattr(dataset, "summarize"):
        print("[main] Calling dataset.summarize()")
        summary["dataset_summary"] = dataset.summarize()
    elif hasattr(dataset, "describe"):
        print("[main] Calling dataset.describe()")
        summary["dataset_summary"] = dataset.describe()
    else:
        print("[main] Dataset has no summarize/describe method")
        summary["dataset_summary"] = {
            "num_pairs": len(dataset_items),
        }

    write_json(summary_json_path, summary)
    write_json(settings_json_path, cfg)

    print(f"[main] Finished run. metrics_csv_path={metrics_csv_path}")
    print(f"[main] summary_json_path={summary_json_path}")
    print(f"[main] settings_json_path={settings_json_path}")


if __name__ == "__main__":
    main()