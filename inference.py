import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from video_clip import MODEL_NAME, build_open_clip, frames_to_tensor, sample_video_frames

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("xgboost is required. Install it via `pip install xgboost`.") from exc

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

logger = logging.getLogger(__name__)


def encode_video_feature(
    video_path: str,
    seq_len: int,
    model,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    frames_bgr = sample_video_frames(video_path, seq_len)
    batch = frames_to_tensor(frames_bgr, preprocess).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
    features = F.normalize(features, dim=-1)
    pooled = features.max(dim=0).values
    pooled = F.normalize(pooled, dim=-1)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def load_summary(summary_path: Optional[Path]) -> dict:
    if summary_path is None:
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Summary {summary_path} does not contain a JSON object.")
    return data


def run_inference(
    video_path: str,
    seq_len: int,
    action_label: str,
    threshold: float,
    display: bool,
    booster: xgb.Booster,
    model,
    preprocess,
    device: torch.device,
    model_path: Path,
) -> dict:
    feature_vec = encode_video_feature(video_path, seq_len, model, preprocess, device)
    dmatrix = xgb.DMatrix(feature_vec.reshape(1, -1))
    prob = float(booster.predict(dmatrix)[0])
    predicted = prob >= threshold

    result = {
        "video": video_path,
        "model": str(model_path),
        "action": action_label,
        "probability": prob,
        "threshold": threshold,
        "prediction": bool(predicted),
    }

    logger.info("Predicted probability %.4f (threshold %.2f) => %s", prob, threshold, predicted)

    if display:
        if not HAS_CV2:
            raise ImportError("OpenCV is required for display. Install opencv-python.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        label_text = f"{action_label}: {prob:.3f}"
        verdict_text = "POSITIVE" if predicted else "NEGATIVE"

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                cv2.rectangle(frame, (8, 8), (8 + 360, 80), (0, 0, 0), thickness=-1)
                cv2.putText(frame, label_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    verdict_text,
                    (16, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if predicted else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                # cv2.imshow("XGBoost Action Prediction", frame)
                # if cv2.waitKey(25) & 0xFF == ord("q"):
                #     break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return result


def expand_video_inputs(paths: Iterable[Path], recursive: bool) -> List[str]:
    video_exts = {".mp4", ".avi", ".mkv", ".mov"}
    collected: List[str] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in video_exts:
                    collected.append(str(candidate))
        elif path.is_file():
            if path.suffix.lower() in video_exts:
                collected.append(str(path))
            else:
                logger.warning("Skipping non-video file %s", path)
        else:
            logger.warning("Skipping missing path %s", path)
    return sorted(collected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on a video using a trained XGBoost action classifier."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Path(s) to video file(s) or directories containing videos.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained XGBoost model (.json/.bin).")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional training summary JSON to auto-fill action label and seq_len.",
    )
    parser.add_argument("--action", type=str, default=None, help="Target action label. Overrides summary if provided.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Number of frames sampled per video. Defaults to value from summary or 16.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive prediction.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for video files.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--pretrained", type=str, default="mobileclip2_s2")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary_data = load_summary(args.summary)

    action_label = args.action or summary_data.get("target_action")
    if action_label is None:
        raise ValueError("Action label must be provided via --action or summary JSON.")

    seq_len = args.seq_len or summary_data.get("seq_len") or 16

    video_paths = expand_video_inputs(args.inputs, args.recursive)
    if not video_paths:
        raise ValueError("No video files found for the given input path(s).")

    booster = xgb.Booster()
    booster.load_model(str(args.model))
    model, _, preprocess, device = build_open_clip(args.model_name, args.pretrained)

    results = []
    for video_path in video_paths:
        result = run_inference(
            video_path=video_path,
            seq_len=seq_len,
            action_label=action_label,
            threshold=args.threshold,
            display=not args.no_display,
            booster=booster,
            model=model,
            preprocess=preprocess,
            device=device,
            model_path=args.model,
        )
        results.append(result)

    output = results[0] if len(results) == 1 else results
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
