import argparse
import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import cv2

from video_clip import MODEL_NAME, build_open_clip, frames_to_tensor, sample_video_frames

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("xgboost is required. Install it via `pip install xgboost`.") from exc

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

logger = logging.getLogger(__name__)


if HAS_TQDM:

    class XGBoostProgressCallback(xgb.callback.TrainingCallback):
        """tqdm-based progress bar for XGBoost training."""

        def __init__(self, total_rounds: int) -> None:
            self.total_rounds = total_rounds
            self.pbar = None

        def before_training(self, model):
            self.pbar = tqdm(total=self.total_rounds, desc="Training XGBoost", unit="iter")
            return model

        def after_iteration(self, model, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]):  # type: ignore[override]
            if self.pbar is not None:
                self.pbar.update(1)
                if evals_log:
                    postfix = {}
                    for dataset_name, metrics in evals_log.items():
                        for metric_name, values in metrics.items():
                            postfix[f"{dataset_name}-{metric_name}"] = f"{values[-1]:.4f}"
                            if len(postfix) >= 3:
                                break
                        if len(postfix) >= 3:
                            break
                    if postfix:
                        self.pbar.set_postfix(postfix, refresh=False)
            return False

        def after_training(self, model) -> xgb.Booster:
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            return model


def iter_video_files(root: Path) -> Dict[str, List[Path]]:
    """Return mapping from action label to the list of video paths."""
    dataset: Dict[str, List[Path]] = defaultdict(list)
    video_exts = (".avi", ".mp4", ".mov", ".mkv")
    for action_dir in root.iterdir():
        if not action_dir.is_dir():
            continue
        for path in action_dir.iterdir():
            if path.suffix.lower() in video_exts:
                dataset[action_dir.name].append(path)
        if not dataset[action_dir.name]:
            logger.warning("No videos found under %s", action_dir)
    return dataset


def encode_video_feature(
    video_path: Path,
    seq_len: int,
    model,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    """Encode a video by uniformly sampling seq_len frames across its full duration."""
    if seq_len <= 0:
        raise ValueError("--seq-len must be positive.")

    frames_bgr = sample_video_frames(str(video_path), seq_len)
    batch = frames_to_tensor(frames_bgr, preprocess).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)  # [seq_len, D]

    features = F.normalize(features, dim=-1)
    top_features = features.max(dim=0).values
    top_features = F.normalize(top_features, dim=-1)
    return top_features.detach().cpu().numpy().astype(np.float32, copy=False)


def collect_features(
    dataset: Dict[str, List[Path]],
    dataset_root: Path,
    target_action: str,
    seq_len: int,
    model,
    preprocess,
    device: torch.device,
    negatives: Optional[Sequence[str]],
    max_samples_per_class: Optional[int],
    seed: int,
    show_progress: bool,
    workers: int,
    feature_cache_root: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Collect per-video feature vectors and labels for XGBoost training."""
    rng = np.random.default_rng(seed)
    selected_actions = set(dataset.keys()) if negatives is None else {target_action, *negatives}
    if target_action not in dataset:
        raise ValueError(f"Action '{target_action}' not found in dataset.")
    if target_action not in selected_actions:
        selected_actions.add(target_action)

    dataset_root = dataset_root.resolve()
    tasks: List[Tuple[str, Path, Path]] = []
    for action_label, videos in dataset.items():
        if action_label not in selected_actions:
            continue
        indices = np.arange(len(videos))
        rng.shuffle(indices)
        if max_samples_per_class is not None and len(indices) > max_samples_per_class:
            indices = indices[: max_samples_per_class]
        for idx in indices:
            video_path = videos[idx]
            try:
                rel_path = video_path.resolve().relative_to(dataset_root)
            except (ValueError, RuntimeError):
                rel_path = Path(video_path.name)
            tasks.append((action_label, video_path, rel_path))

    if not tasks:
        raise RuntimeError("No matching videos were collected. Check dataset paths and parameters.")

    order = rng.permutation(len(tasks))
    tasks = [tasks[i] for i in order]
    indexed_tasks: List[Tuple[int, str, Path, Path]] = [
        (idx, action_label, video_path, rel_path) for idx, (action_label, video_path, rel_path) in enumerate(tasks)
    ]

    if show_progress and not HAS_TQDM:
        logger.warning("tqdm is not installed; progress bars are disabled.")

    progress_bar = (
        tqdm(total=len(tasks), desc="Encoding videos", unit="video")
        if show_progress and HAS_TQDM
        else None
    )

    feature_store: Dict[int, np.ndarray] = {}
    action_store: Dict[int, str] = {}
    video_store: Dict[int, str] = {}
    ignored_store: Dict[int, tuple[str, str, int]] = {}
    ignore_lock = Lock()

    def encode_task(
        index: int,
        action_label: str,
        video_path: Path,
        rel_path: Path,
    ) -> Tuple[int, Optional[np.ndarray], Optional[str], Path]:
        cache_file: Optional[Path] = None
        frame_count: Optional[int] = None

        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to inspect %s: %s", video_path, exc)

        if frame_count is not None and frame_count > 0 and frame_count < seq_len:
            with ignore_lock:
                ignored_store[index] = (action_label, str(rel_path), frame_count)
            return index, None, None, video_path

        if feature_cache_root is not None:
            cache_file = feature_cache_root / rel_path.with_suffix(".npy")
            try:
                if cache_file.exists():
                    feature_vec = np.load(cache_file, allow_pickle=False)
                    feature_vec = np.asarray(feature_vec, dtype=np.float32)
                    return index, feature_vec, action_label, video_path
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to read cache %s: %s; recomputing", cache_file, exc)

        try:
            feature_vec = encode_video_feature(video_path, seq_len, model, preprocess, device)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to encode %s: %s", video_path, exc)
            return index, None, None, video_path

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, feature_vec, allow_pickle=False)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to write cache %s: %s", cache_file, exc)

        return index, feature_vec, action_label, video_path

    worker_count = max(1, workers)
    if worker_count == 1:
        for index, action_label, video_path, rel_path in indexed_tasks:
            idx, feature_vec, label_action, _video_path = encode_task(index, action_label, video_path, rel_path)
            if feature_vec is None or label_action is None:
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
            feature_store[idx] = feature_vec
            action_store[idx] = label_action
            video_store[idx] = str(video_path)
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(encode_task, index, action_label, video_path, rel_path): index
                for index, action_label, video_path, rel_path in indexed_tasks
            }
            for future in as_completed(future_map):
                idx, feature_vec, label_action, video_path = future.result()
                if feature_vec is not None and label_action is not None:
                    feature_store[idx] = feature_vec
                    action_store[idx] = label_action
                    video_store[idx] = str(video_path)
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    if not feature_store:
        raise RuntimeError("No features were collected. Check dataset paths and parameters.")

    if ignored_store and feature_cache_root is not None:
        ignore_dir = feature_cache_root.parent
        ignore_dir.mkdir(parents=True, exist_ok=True)
        ignore_csv_path = ignore_dir / f"data_ignore_seq_{seq_len}.csv"
        ignored_rows = sorted(ignored_store.values(), key=lambda row: row[1])
        with ignore_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["action", "video", "frame_count"])
            writer.writerows(ignored_rows)
        logger.info(
            "Excluded %d video(s) shorter than %d frames. Saved list to %s",
            len(ignored_rows),
            seq_len,
            ignore_csv_path,
        )

    ordered_indices = sorted(feature_store.keys())
    features = [feature_store[idx] for idx in ordered_indices]
    labels = [1 if action_store[idx] == target_action else 0 for idx in ordered_indices]
    metadata = [{"action": action_store[idx], "video": video_store[idx]} for idx in ordered_indices]

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    return X, y, metadata


def try_split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    valid_ratio: float,
    seed: int,
    max_attempts: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Split dataset into train/validation while keeping both classes present."""
    num_samples = X.shape[0]
    if num_samples < 2 or valid_ratio <= 0.0:
        logger.info("Skipping validation split (valid_ratio <= 0 or insufficient samples).")
        return X, y, None, None, np.arange(num_samples), np.array([], dtype=int)

    valid_ratio = min(max(valid_ratio, 0.0), 0.5)
    num_valid = int(round(num_samples * valid_ratio))
    num_valid = max(1, min(num_valid, num_samples - 1))

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        perm = rng.permutation(num_samples)
        valid_idx = perm[:num_valid]
        train_idx = perm[num_valid:]

        train_labels = y[train_idx]
        valid_labels = y[valid_idx]

        if train_labels.size == 0:
            continue
        if train_labels.sum() in (0, train_labels.size):
            continue
        if valid_labels.size > 0 and valid_labels.sum() in (0, valid_labels.size) and valid_labels.size > 1:
            continue

        return (
            X[train_idx],
            y[train_idx],
            X[valid_idx] if valid_labels.size > 0 else None,
            y[valid_idx] if valid_labels.size > 0 else None,
            train_idx,
            valid_idx,
        )

    logger.warning("Unable to create stratified validation split; proceeding without validation.")
    return X, y, None, None, np.arange(num_samples), np.array([], dtype=int)


def parse_eval_result(raw: str) -> Dict[str, float]:
    """Parse XGBoost eval result string into a dict."""
    metrics: Dict[str, float] = {}
    for token in raw.strip().split():
        if token.startswith("["):
            continue
        if ":" not in token:
            continue
        key, value = token.split(":", maxsplit=1)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def compute_binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute threshold-based binary metrics."""
    preds = (probs >= 0.5).astype(np.int32)
    accuracy = float((preds == labels).mean())

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": 0.5,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    params: Dict[str, float],
    num_rounds: int,
    early_stopping_rounds: Optional[int],
    show_progress: bool,
) -> Tuple[xgb.Booster, xgb.DMatrix, Optional[xgb.DMatrix]]:
    """Train an XGBoost model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]

    dvalid = None
    if X_valid is not None and y_valid is not None and X_valid.size > 0:
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        evals.append((dvalid, "valid"))

    callbacks = []
    if show_progress and HAS_TQDM:
        callbacks.append(XGBoostProgressCallback(num_rounds))
    elif show_progress and not HAS_TQDM:
        logger.warning("tqdm is not installed; training progress bar disabled.")

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=False,
        callbacks=callbacks,
    )

    logger.info(
        "Training finished at iteration %d (best score %.4f)",
        booster.best_iteration if booster.best_iteration is not None else num_rounds,
        booster.best_score if booster.best_score is not None else float("nan"),
    )
    return booster, dtrain, dvalid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier for an action in UCF101 using MobileCLIP features."
    )
    parser.add_argument("ucf101_dir", type=Path, help="Path to the UCF101 dataset root.")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="Action class treated as the positive label.",
    )
    parser.add_argument(
        "--negatives",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of action classes used as negatives (defaults to all other classes).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="Total frames to sample per video when encoding features.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap on videos per class to balance training data.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Portion of data held out for validation.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for sampling and splitting.")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--pretrained", type=str, default="mobileclip2_s2")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--xgb-rounds", type=int, default=200, help="Number of boosting rounds.")
    parser.add_argument("--xgb-depth", type=int, default=8, help="Max depth for XGBoost trees.")
    parser.add_argument("--xgb-eta", type=float, default=0.03, help="Learning rate for XGBoost.")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="Row subsample ratio per tree.")
    parser.add_argument("--xgb-colsample", type=float, default=0.8, help="Column subsample ratio per tree.")
    parser.add_argument("--xgb-alpha", type=float, default=0.01, help="L1 regularization term (reg_alpha).")
    parser.add_argument("--xgb-lambda", type=float, default=1.0, help="L2 regularization term (reg_lambda).")
    parser.add_argument("--xgb-min-child-weight", type=float, default=3, help="XGB min_child_weight.")
    parser.add_argument(
        "--xgb-scale-pos-weight",
        type=float,
        default=None,
        help="Positive class weight (scale_pos_weight). Defaults to negatives/positives ratio.",
    )
    parser.add_argument("--early-stopping", type=int, default=30, help="Early stopping rounds (requires validation).")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Path to save the trained XGBoost model (.json or .bin). Defaults to models/<action>_<seq_len>_xgb.json.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path to save training summary JSON. Defaults to models/<action>_<seq_len>_summary.json.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during feature extraction and training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for feature extraction.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info(
        "Training parameters: action=%s negatives=%s seq_len=%d max_samples_per_class=%s valid_ratio=%.2f seed=%d"
        " model=%s pretrained=%s xgb_rounds=%d xgb_depth=%d xgb_eta=%.3f xgb_subsample=%.2f xgb_colsample=%.2f"
        " xgb_alpha=%.3f xgb_lambda=%.3f xgb_min_child_weight=%.2f scale_pos_weight=%s workers=%d",
        args.action,
        args.negatives,
        args.seq_len,
        args.max_samples_per_class,
        args.valid_ratio,
        args.seed,
        args.model,
        args.pretrained,
        args.xgb_rounds,
        args.xgb_depth,
        args.xgb_eta,
        args.xgb_subsample,
        args.xgb_colsample,
        args.xgb_alpha,
        args.xgb_lambda,
        args.xgb_min_child_weight,
        args.xgb_scale_pos_weight,
        args.workers,
    )

    if not args.ucf101_dir.is_dir():
        raise FileNotFoundError(f"{args.ucf101_dir} is not a directory.")

    default_model_dir = Path("models")
    if args.model_output is None:
        default_model_dir.mkdir(parents=True, exist_ok=True)
        args.model_output = default_model_dir / f"{args.action}_{args.seq_len}_xgb.json"
    else:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_output is None:
        args.summary_output = default_model_dir / f"{args.action}_{args.seq_len}_summary.json"
    else:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    model, _, preprocess, device = build_open_clip(args.model, args.pretrained)
    dataset = iter_video_files(args.ucf101_dir)
    if not dataset:
        raise RuntimeError(f"No actions discovered under {args.ucf101_dir}")

    cache_dir = Path("cache")
    cache_path = cache_dir / f"ucf101_{args.seq_len}_seq.npy"
    worker_count = max(1, args.workers)
    expected_params = {
        "dataset": str(args.ucf101_dir.resolve()),
        "action": args.action,
        "negatives": sorted(args.negatives) if args.negatives else None,
        "seq_len": args.seq_len,
        "max_samples_per_class": args.max_samples_per_class,
        "seed": args.seed,
        "workers": worker_count,
        "xgb_depth": args.xgb_depth,
        "xgb_eta": args.xgb_eta,
        "xgb_subsample": args.xgb_subsample,
        "xgb_colsample": args.xgb_colsample,
        "xgb_scale_pos_weight": args.xgb_scale_pos_weight,
    }

    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    metadata: Optional[List[dict]] = None

    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=True).item()
            if payload.get("params") == expected_params:
                X = np.asarray(payload.get("features"), dtype=np.float32)
                y = np.asarray(payload.get("labels"), dtype=np.int32)
                metadata = payload.get("metadata", [])
                logger.info("Loaded cached features from %s", cache_path)
            else:
                logger.info("Cache parameters mismatch for %s; regenerating.", cache_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load cache %s: %s", cache_path, exc)

    if X is None or y is None:
        feature_cache_root = cache_dir / "features" / f"{args.seq_len}"
        X, y, metadata = collect_features(
            dataset=dataset,
            dataset_root=args.ucf101_dir,
            target_action=args.action,
            seq_len=args.seq_len,
            model=model,
            preprocess=preprocess,
            device=device,
            negatives=args.negatives,
            max_samples_per_class=args.max_samples_per_class,
            seed=args.seed,
            show_progress=HAS_TQDM and not args.no_progress,
            workers=worker_count,
            feature_cache_root=feature_cache_root,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": expected_params,
            "features": X,
            "labels": y,
            "metadata": metadata,
        }
        np.save(cache_path, payload, allow_pickle=True)
        logger.info("Cached features to %s", cache_path)

    total_positive = int(y.sum())
    total_negative = int(y.size - total_positive)
    if total_positive == 0 or total_negative == 0:
        raise RuntimeError(
            "Training requires both positive and negative samples. Adjust --negatives or dataset selection."
        )

    X_train, y_train, X_valid, y_valid, train_idx, valid_idx = try_split_dataset(
        X, y, valid_ratio=args.valid_ratio, seed=args.seed
    )

    train_pos = int(y_train.sum())
    train_neg = int(y_train.size - train_pos)
    if train_pos == 0 or train_neg == 0:
        raise RuntimeError(
            "Training split lacks class diversity. Adjust --valid-ratio or sampling parameters."
        )

    scale_pos_weight = (
        args.xgb_scale_pos_weight
        if args.xgb_scale_pos_weight is not None
        else max(total_negative / total_positive, 1.0)
    )

    logger.info(
        "Using scale_pos_weight=%.3f (positives=%d, negatives=%d)",
        scale_pos_weight,
        total_positive,
        total_negative,
    )

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": args.xgb_eta,
        "max_depth": args.xgb_depth,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "seed": args.seed,
        "nthread": -1,
        "min_child_weight": args.xgb_min_child_weight,
        "reg_alpha": args.xgb_alpha,
        "reg_lambda": args.xgb_lambda,
        "scale_pos_weight": scale_pos_weight,
    }

    booster, dtrain, dvalid = train_xgboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=xgb_params,
        num_rounds=args.xgb_rounds,
        early_stopping_rounds=args.early_stopping,
        show_progress=not (args.no_progress is True)
    )

    train_probs = booster.predict(dtrain)
    train_metrics = compute_binary_metrics(train_probs, y_train)
    train_metrics.update(parse_eval_result(booster.eval(dtrain)))

    valid_metrics = None
    if dvalid is not None and y_valid is not None and y_valid.size > 0:
        valid_probs = booster.predict(dvalid)
        valid_metrics = compute_binary_metrics(valid_probs, y_valid)
        valid_metrics.update(parse_eval_result(booster.eval(dvalid)))

    summary = {
        "target_action": args.action,
        "total_samples": int(y.size),
        "train_samples": int(y_train.size),
        "valid_samples": int(y_valid.size if y_valid is not None else 0),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "train_positive": train_pos,
        "train_negative": train_neg,
        "feature_dim": int(X.shape[1]),
        "seq_len": args.seq_len,
        "xgb_params": {
            "eta": args.xgb_eta,
            "max_depth": args.xgb_depth,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample,
            "reg_alpha": args.xgb_alpha,
            "reg_lambda": args.xgb_lambda,
            "min_child_weight": args.xgb_min_child_weight,
            "scale_pos_weight": scale_pos_weight,
        },
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_output": str(args.model_output) if args.model_output else None,
    }

    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(args.model_output))
        logger.info("Saved XGBoost model to %s", args.model_output)

    serialized = json.dumps(summary, indent=2)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(serialized, encoding="utf-8")
        logger.info("Saved training summary to %s", args.summary_output)

    print(serialized)


if __name__ == "__main__":
    main()
