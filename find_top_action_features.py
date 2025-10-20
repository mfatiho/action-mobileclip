import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from video_clip import MODEL_NAME, build_open_clip, frames_to_tensor, sample_video_frames

logger = logging.getLogger(__name__)


def iter_video_files(root: Path, action: Optional[str]) -> Dict[str, List[Path]]:
    """Return mapping from action label to the list of video paths."""
    if action is not None:
        action_dirs = [root / action]
    else:
        action_dirs = [p for p in root.iterdir() if p.is_dir()]

    dataset: Dict[str, List[Path]] = defaultdict(list)
    video_exts = (".avi", ".mp4", ".mov", ".mkv")
    for action_dir in action_dirs:
        if not action_dir.is_dir():
            logger.warning("Skipping missing action directory %s", action_dir)
            continue
        for path in action_dir.iterdir():
            if path.suffix.lower() in video_exts:
                dataset[action_dir.name].append(path)
        if not dataset[action_dir.name]:
            logger.warning("No videos found under %s", action_dir)
    return dataset


def encode_video_top_features(
    video_path: Path,
    seq_len: int,
    model,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """Encode a video by uniformly sampling seq_len frames across its full duration."""
    if seq_len <= 0:
        raise ValueError("--seq-len must be positive.")

    frames_bgr = sample_video_frames(str(video_path), seq_len)
    batch = frames_to_tensor(frames_bgr, preprocess).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)  # [T, D] where T is sampled frames

    features = F.normalize(features, dim=-1)
    top_features = features.max(dim=0).values
    top_features = F.normalize(top_features, dim=-1)
    return top_features


def aggregate_action_features(
    action: str,
    videos: List[Path],
    seq_len: int,
    model,
    preprocess,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Aggregate features for the given action over all provided videos."""
    if not videos:
        return None

    action_features: List[torch.Tensor] = []
    for video_path in videos:
        try:
            feature_vec = encode_video_top_features(
                video_path=video_path,
                seq_len=seq_len,
                model=model,
                preprocess=preprocess,
                device=device,
            )
            action_features.append(feature_vec)
            logger.debug("Encoded %s for action %s", video_path.name, action)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to encode %s: %s", video_path, exc)

    if not action_features:
        logger.warning("No encodings produced for action %s", action)
        return None

    stacked = torch.stack(action_features, dim=0)
    max_features = stacked.max(dim=0).values
    max_features = F.normalize(max_features, dim=-1)
    return max_features


def compute_top_k(features: torch.Tensor, k: int) -> List[dict]:
    """Return list of top-k feature indices and values."""
    k = min(k, features.numel())
    values, indices = torch.topk(features, k=k, largest=True)
    return [
        {"feature_index": int(idx.item()), "value": float(val.item())}
        for idx, val in zip(indices, values)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find top MobileCLIP feature dimensions for UCF101 actions."
    )
    parser.add_argument("ucf101_dir", type=Path, help="Path to the UCF101 dataset root.")
    parser.add_argument(
        "--action",
        type=str,
        default=None,
        help="Action class to process. If omitted, all action directories are processed.",
    )
    parser.add_argument("--seq-len", type=int, default=16, help="Total frames to sample per video.")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--pretrained", type=str, default="mobileclip2_s2")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to persist the JSON results.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.ucf101_dir.is_dir():
        raise FileNotFoundError(f"{args.ucf101_dir} is not a directory.")

    model, _, preprocess, device = build_open_clip(args.model, args.pretrained)

    dataset = iter_video_files(args.ucf101_dir, args.action)
    if not dataset:
        logger.error("No actions were discovered under %s", args.ucf101_dir)
        return

    results = {}
    for action_label, videos in dataset.items():
        logger.info("Processing action %s with %d videos", action_label, len(videos))
        features = aggregate_action_features(
            action=action_label,
            videos=videos,
            seq_len=args.seq_len,
            model=model,
            preprocess=preprocess,
            device=device,
        )
        if features is None:
            continue
        top_features = compute_top_k(features, args.seq_len)
        results[action_label] = {
            "top_features": top_features,
            "num_videos": len(videos),
        }

    serialized = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        logger.info("Saved results to %s", args.output)

    print(serialized)


if __name__ == "__main__":
    main()
