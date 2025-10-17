# action-mobileclip

Minimal OpenCLIP video-to-text classification utility.

## Install
```bash
pip install torch open-clip-torch opencv-python pillow
```

## Usage
```bash
# Single prompt (fallback)
python video_open_clip_minimal.py path/to/video.mp4 "kedi"

# Multiple prompts (recommended)
python video_open_clip_minimal.py path/to/video.mp4 "kedi" --texts "kedi" "köpek" "araba" --frames 8 --model ViT-B-32 --pretrained laion2b_s34b_b79k
```

- Video’dan `--frames` adet frame örneklenir, image-encoder’dan geçirilir ve özellikler ortalanır.
- Metin(ler) text-encoder’dan geçirilir ve normalize edilir.
- CLIP tarzı benzerlikten `softmax` ile olasılıklar üretilir ve yazdırılır.
- `combined_features` de ayrıca döner (image_avg ile text ortalamasının concat’i) – ileri kullanım için.

## Output (stdout)
```json
{"texts": ["kedi", "köpek", "araba"], "probs": [0.72, 0.20, 0.08]}
```

## Notes
- CUDA varsa otomatik mixed precision açıktır; kapatmak için `--no-amp`.
- Modeller: `--model` ve `--pretrained` ile seçilebilir (OpenCLIP isimleri).