import argparse
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:  # pragma: no cover
    HAS_CV2 = False


def sample_video_frames(video_path: str, num_frames: int) -> List["cv2.Mat"]:
    if not HAS_CV2:
        raise ImportError("OpenCV (opencv-python) is required for video reading.")
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
            continue
        frames.append(frame_bgr)
    cap.release()

    if not frames:
        raise RuntimeError("Failed to read any frames from video.")
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
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: Optional[torch.device] = None,
):
    import open_clip

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, tokenizer, preprocess, device


def encode_video_and_text(
    video_path: str,
    text: Sequence[str] | str,
    num_frames: int = 8,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
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

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    with torch.no_grad():
        if device.type == "cuda":
            ctx = torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype)
        else:
            ctx = torch.autocast(device_type="cpu", enabled=False)
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
            scale = torch.tensor(1.0, device=device)
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


if __name__ == "__main__":
    # Minimal CLI usage
    # pip install open-clip-torch opencv-python pillow
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("prompt", type=str, help="Text prompt (fallback if --texts not given)")
    parser.add_argument("--texts", type=str, nargs="*", default=None, help="One or more text prompts")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    args = parser.parse_args()

    texts_cli = args.texts if args.texts is not None else [args.prompt]

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
