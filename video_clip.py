import argparse
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, List, Optional, Sequence

import torch
import torch.nn.functional as F

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from open_clip import factory as open_clip_factory
from mobileclip_loader import load_mobileclip2_s2

logger = logging.getLogger(__name__)
import cv2
HAS_CV2 = True

MODEL_NAME = "MobileCLIP2-S2"

def sample_video_frames(video_path: str, num_frames: int) -> List["cv2.Mat"]:
    if not HAS_CV2:
        raise ImportError("OpenCV (opencv-python) is required for video reading.")
    logger.debug("Opening %s to sample %d frames", video_path, num_frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video has no frames or frame count unavailable.")

    # Uniformly sample indices in [0, frame_count-1]
    indices = [min(int(round(i * (frame_count - 1) / max(1, num_frames - 1))), frame_count - 1) for i in range(num_frames)]

    frames: List["cv2.Mat"] = []
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            logger.warning("Failed to read frame %d from %s", target_idx, video_path)
            continue
        frames.append(frame_bgr)
    cap.release()

    if not frames:
        raise RuntimeError("Failed to read any frames from video.")
    logger.debug("Sampled %d frames from %s", len(frames), video_path)
    return frames


def frames_to_tensor(frames_bgr: List["cv2.Mat"], preprocess) -> torch.Tensor:
    # open_clip preprocess expects PIL Image in RGB
    from PIL import Image

    processed: List[torch.Tensor] = []
    for frame_bgr in frames_bgr:
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)
        processed.append(preprocess(image))
    batch = torch.stack(processed, dim=0)
    return batch


def build_open_clip(
    model_name: str = MODEL_NAME,
    pretrained: str = "mobileclip2_s2",
    device: Optional[torch.device] = None,
):
    _ = model_name
    _ = pretrained
    model, preprocess, tokenizer, resolved_device = load_mobileclip2_s2(device=device)
    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is None:
        model.logit_scale = torch.tensor(1.0, device=resolved_device)
    logger.info("Loaded %s on %s", MODEL_NAME, resolved_device)
    return model, tokenizer, preprocess, resolved_device


def encode_video_and_text(
    video_path: str,
    text: Sequence[str] | str,
    num_frames: int = 8,
    model_name: str = MODEL_NAME,
    pretrained: str = "mobileclip2_s2",
    use_amp: bool = True,
) -> dict:
    model, tokenizer, preprocess, device = build_open_clip(model_name, pretrained)

    frames_bgr = sample_video_frames(video_path, num_frames)
    image_batch = frames_to_tensor(frames_bgr, preprocess).to(device)

    if isinstance(text, str):
        texts: List[str] = [text]
    else:
        texts = list(text)

    tokens = tokenizer(texts).to(device)
    logger.info(
        "Encoding %d prompt(s) against %d sampled frames from %s",
        len(texts),
        len(frames_bgr),
        video_path,
    )

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    with torch.no_grad():
        if device.type == "cuda":
            ctx = torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype)
        else:
            ctx = torch.amp.autocast("cpu", enabled=False)
        with ctx:
            image_features_per_frame = model.encode_image(image_batch)
            text_features = model.encode_text(tokens)

        # Average image features across frames and L2-normalize
        image_features_avg = image_features_per_frame.mean(dim=0, keepdim=True)
        image_features_avg = F.normalize(image_features_avg, dim=-1)
        # Normalize text features (CLIP convention)
        text_features = F.normalize(text_features, dim=-1)
        # Concatenate features (kept for downstream usage)
        combined_features = torch.cat([image_features_avg, text_features.mean(dim=0, keepdim=True)], dim=-1)

        # CLIP-style logits and probabilities over provided texts
        # Use model.logit_scale if present; default to 1.0 otherwise
        logit_scale = getattr(model, "logit_scale", None)
        if logit_scale is not None:
            scale = logit_scale.exp()
        else:
            scale = torch.tensor(100.0, device=device)
        logits_per_image = scale * image_features_avg @ text_features.t()  # [1, num_texts]
        probs = logits_per_image.softmax(dim=-1)  # probabilities over texts

    return {
        "texts": texts,
        "image_features_per_frame": image_features_per_frame,
        "image_features_avg": image_features_avg,
        "text_features": text_features,
        "combined_features": combined_features,
        "logits_per_image": logits_per_image,
        "probs_per_image": probs,
    }


def stream_and_display(
    source: str | int,
    texts: Sequence[str],
    num_frames: int,
    model_name: str,
    pretrained: str,
    use_amp: bool,
    window_name: str = "OpenCLIP Stream",
) -> None:
    if not HAS_CV2:
        raise ImportError("OpenCV (opencv-python) is required for video streaming.")

    model, tokenizer, preprocess, device = build_open_clip(model_name, pretrained)

    # Precompute text features once
    tokens = tokenizer(list(texts)).to(device)
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32
    logger.info(
        "Streaming from %s with %d prompt(s) and window size %d",
        source,
        len(texts),
        num_frames,
    )

    with torch.no_grad():
        if device.type == "cuda":
            text_ctx = torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype)
        else:
            text_ctx = torch.amp.autocast("cpu", enabled=False)
        with text_ctx:
            text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    logit_scale = getattr(model, "logit_scale", None)
    scale = logit_scale.exp() if logit_scale is not None else torch.tensor(100.0, device=device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    
    # Sliding window over per-frame features
    feature_queue: Deque[torch.Tensor] = deque()
    sum_features: Optional[torch.Tensor] = None  # shape [1, D]
    current_top: Optional[tuple[str, float]] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Encode single frame to a feature vector (1, D)
            frame_tensor = frames_to_tensor([frame], preprocess).to(device)
            with torch.no_grad():
                if device.type == "cuda":
                    ctx = torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype)
                else:
                    ctx = torch.amp.autocast("cpu", enabled=False)
                with ctx:
                    feat = model.encode_image(frame_tensor)  # [1, D]

            # Update queue and running sum (unnormalized features)
            feature_queue.append(feat)
            if sum_features is None:
                sum_features = feat.clone()
            else:
                sum_features = sum_features + feat

            if len(feature_queue) > num_frames:
                oldest = feature_queue.popleft()
                sum_features = sum_features - oldest

            # Compute probabilities when window is full
            if len(feature_queue) == num_frames:
                assert sum_features is not None
                image_features_avg = sum_features / float(num_frames)  # [1, D]
                image_features_avg = F.normalize(image_features_avg, dim=-1)
                logits = scale * (image_features_avg @ text_features.t())  # [1, N]
                probs = logits.softmax(dim=-1)[0]
                top_idx = int(torch.argmax(probs).item())
                current_top = (texts[top_idx], float(probs[top_idx].item()))
                logger.debug(
                    "Top prediction %s with probability %.3f",
                    current_top[0],
                    current_top[1],
                )

            # Calculate current playback time (seconds)
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                seconds = float(pos_msec) / 1000.0
            else:
                # Fallback to frame index / fps
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0
                seconds = (frame_idx / fps) if fps > 1e-3 else 0.0

            # Overlay current prediction
            if current_top is not None:
                label, p = current_top
                text = f"{label}: {p:.2f}"
                cv2.rectangle(frame, (8, 8), (8 + 8 + 6 * len(text), 40), (0, 0, 0), thickness=-1)
                cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                time_str = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
                logger.info("Prediction=%s prob=%.3f time=%s", label, p, time_str)
            # cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Minimal CLI usage
    # pip install open-clip-torch opencv-python pillow
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Video path or camera index (e.g., 0)")
    parser.add_argument("prompt", type=str, help="Text prompt (fallback if --texts not given)")
    parser.add_argument("--texts", type=str, nargs="*", default=None, help="One or more text prompts")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames per chunk")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--pretrained", type=str, default="mobileclip2_s2")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--stream", action="store_true", help="Process as a live stream and display online")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    texts_cli = args.texts if args.texts is not None else [args.prompt]

    if args.stream:
        # Allow camera index
        try:
            src: str | int = int(args.video)
        except ValueError:
            src = args.video
        stream_and_display(
            source=src,
            texts=texts_cli,
            num_frames=args.frames,
            model_name=args.model,
            pretrained=args.pretrained,
            use_amp=not args.no_amp,
        )
    else:
        outputs = encode_video_and_text(
            video_path=args.video,
            text=texts_cli,
            num_frames=args.frames,
            model_name=args.model,
            pretrained=args.pretrained,
            use_amp=not args.no_amp,
        )
        probs = outputs["probs_per_image"].float().cpu().numpy().tolist()[0]
        print({"texts": outputs["texts"], "probs": probs})
