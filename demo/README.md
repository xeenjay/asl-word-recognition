# Presentation demo video

`make_demo_video.py` renders the ~30-second presentation video. Sign clips play on the left. On the right, the four systems' Top-1 answers appear and are marked with a green check and chime (correct) or a red cross and buzzer (wrong). An end card shows the final test Top-1 of the four systems.

Requirements: `ffmpeg` (with libx264 and AAC) on the PATH, NumPy, and Pillow (`python -m pip install pillow`).

```bash
python demo/make_demo_video.py demo/preview_spec.json outputs/demo-preview.mp4
```

`preview_spec.json` is a style preview. Its clips are placeholders and its "EXAMPLE" labels are fake. The end-card values are the real one-shot test Top-1 results from [Results](../docs/RESULTS.md).

For the real video, write a spec with the same structure:

- `video`: the clip's path, relative to the spec file.
- `truth`: the exact dataset gloss.
- `predictions`: each system's Top-1, in the order kNN, Linear SVM, Random Forest, PoET. Copy them from saved model outputs; never type in or re-run predictions. Correctness is computed by exact string match.
- Remove `watermark`.

Use validation clips only, and keep the clips and the rendered video out of Git: the ASL Citizen license does not allow redistributing the data. `outputs/` is ignored. Fonts are found automatically on Linux, macOS, and Windows; set `DEMO_FONT` and `DEMO_FONT_BOLD` to `.ttf` files to override.
