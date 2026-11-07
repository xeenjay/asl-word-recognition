#!/usr/bin/env python3
"""Render a short presentation video: sign clips with four systems' answers.

Each clip plays (looping) on the left. On the right, the four systems' Top-1
predictions appear one by one, then the ground truth, then each answer turns
green with a check and a chime (correct) or red with a cross and a buzzer
(wrong). Correctness is computed from the spec, never typed in by hand.

The predictions must come from saved model outputs. This script only draws
what the spec says; it does not run any model.

    python make_demo_video.py spec.json out.mp4

Spec (JSON):
    {"title": "...", "subtitle": "...", "course": "...", "caption": "...",
     "watermark": null or "text shown on every frame",
     "methods": [{"name": "kNN", "note": "hand-crafted features"}, ... 4 entries],
     "clips": [{"video": null or "path/to/clip.mp4", "trim": [start_s, end_s] (optional),
                "placeholder": "text for a null video", "truth": "GLOSS",
                "predictions": ["GLOSS", ... one per method, in method order]}],
     "summary": optional end card {"title": "...", "subtitle": "...",
                "values": [percent per method, in method order], "footnotes": ["...", ...]}}
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SS, SR = 1920, 1080, 30, 2, 48000   # output size, frame rate, supersampling, audio rate
FONT_CANDIDATES = {   # first existing file wins (Linux, macOS, Windows); DEMO_FONT / DEMO_FONT_BOLD override
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"],
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
}

C = dict(bg=(246, 247, 249), card=(255, 255, 255), border=(224, 228, 234), text=(24, 32, 44),
         muted=(91, 102, 117), faint=(152, 162, 179), green=(22, 163, 74), green_text=(21, 128, 61),
         green_bg=(233, 248, 239), red=(220, 38, 38), red_text=(185, 28, 28), red_bg=(254, 237, 237),
         accent=(37, 99, 235), amber=(180, 83, 9), video=(232, 235, 240), stripe=(224, 228, 234),
         shadow=(15, 23, 42))

# Layout in output pixels.
VIDEO = (90, 150, 1050, 870)              # 4:3 clip area
PANEL_X0, PANEL_X1 = 1120, 1830
TRUTH = (PANEL_X0, 150, PANEL_X1, 282)
ROW_TOP, ROW_H, ROW_GAP = 302, 124, 24
PROGRESS_Y = 886
PRED_X0 = PANEL_X0 + 300                    # predictions never cross into the method column

# End card: one series, so one hue (categorical slot 1, #2a78d6). Checked with the
# dataviz palette validator against the page background and the white card.
BAR = (42, 120, 214)
CARD = (240, 140, 1680, 900)
CHART_X0, CHART_X1 = 660, 1560            # 0% and 100%
BAR_TOP, BAR_STEP, BAR_HALF = 372, 96, 12  # 24px bars, <= the 24px mark cap

# Timing in seconds.
TITLE_S = 2.0
PRED_START, PRED_STEP, PRED_ANIM = 2.2, 0.22, 0.35
TRUTH_AT, TRUTH_ANIM = PRED_START + 3 * PRED_STEP + PRED_ANIM + 0.25, 0.35
MARK_START, MARK_STEP, MARK_ANIM = TRUTH_AT + 0.5, 0.38, 0.32
SEGMENT_S = MARK_START + 3 * MARK_STEP + MARK_ANIM + 0.75
SUMMARY_S = 3.2
FADE_S = 0.22
PLACEHOLDER_LOOP_S = 2.2


def s(value):
    return int(round(value * SS))


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_back(x):
    x = clamp01(x)
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


@lru_cache(maxsize=None)
def font_file(bold):
    override = os.environ.get("DEMO_FONT_BOLD" if bold else "DEMO_FONT")
    for path in ([override] if override else []) + FONT_CANDIDATES[bold]:
        if Path(path).is_file():
            return path
    raise FileNotFoundError("No usable font found; set DEMO_FONT and DEMO_FONT_BOLD to .ttf files")


@lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(font_file(bold), s(size))


def fitted_font(draw, text, max_width, size, bold=True, minimum=22):
    while size > minimum and draw.textlength(text, font=font(size, bold)) > s(max_width):
        size -= 2
    return font(size, bold)


def box(b):
    return tuple(s(v) for v in b)


# ---------------------------------------------------------------- static layers

def static_layer(spec, main):
    image = Image.new("RGB", (s(W), s(H)), C["bg"])
    if main:
        mask = Image.new("L", image.size, 0)
        shade = ImageDraw.Draw(mask)
        rows = [(PANEL_X0, ROW_TOP + i * (ROW_H + ROW_GAP), PANEL_X1, ROW_TOP + i * (ROW_H + ROW_GAP) + ROW_H) for i in range(4)]
        for x0, y0, x1, y1 in [TRUTH, VIDEO, *rows]:
            shade.rounded_rectangle(box((x0, y0 + 5, x1, y1 + 5)), radius=s(18), fill=40)
        image.paste(Image.new("RGB", image.size, C["shadow"]), (0, 0), mask.filter(ImageFilter.GaussianBlur(s(9))))
    draw = ImageDraw.Draw(image)
    if main:
        draw.text((s(90), s(78)), spec["title"], font=font(38, True), fill=C["text"], anchor="lm")
        draw.text((s(90), s(118)), spec.get("subtitle", ""), font=font(24), fill=C["muted"], anchor="lm")
        draw.text((s(W / 2), s(975)), spec.get("caption", ""), font=font(24), fill=C["faint"], anchor="mm")
        draw.rectangle(box((VIDEO[0], PROGRESS_Y, VIDEO[2], PROGRESS_Y + 8)), fill=C["border"])
    if spec.get("watermark"):
        text = spec["watermark"]
        width = draw.textlength(text, font=font(22, True)) / SS + 48
        x0 = W / 2 - width / 2
        y0 = 1032
        draw.rounded_rectangle(box((x0, y0 - 2, x0 + width, y0 + 34)), radius=s(18), outline=C["amber"], width=s(2))
        draw.text((s(W / 2), s(y0 + 16)), text, font=font(22, True), fill=C["amber"], anchor="mm")
    return image


@lru_cache(maxsize=1)
def stripe_tile():
    w, h = VIDEO[2] - VIDEO[0], VIDEO[3] - VIDEO[1]
    tile = Image.new("RGB", (s(w) + s(120), s(h)), C["video"])
    draw = ImageDraw.Draw(tile)
    for x in range(-s(h), tile.width + s(h), s(60)):
        draw.line([(x, s(h)), (x + s(h), 0)], fill=C["stripe"], width=s(18))
    return tile


# ---------------------------------------------------------------- dynamic parts

def draw_placeholder(image, label, t):
    w, h = VIDEO[2] - VIDEO[0], VIDEO[3] - VIDEO[1]
    offset = int((t * s(40)) % s(60))
    image.paste(stripe_tile().crop((offset, 0, offset + s(w), s(h))), (s(VIDEO[0]), s(VIDEO[1])))
    draw = ImageDraw.Draw(image)
    # A dot tracing a figure eight stands in for hand motion.
    phase = 2 * math.pi * (t % PLACEHOLDER_LOOP_S) / PLACEHOLDER_LOOP_S
    cx = VIDEO[0] + w / 2 + 170 * math.sin(phase)
    cy = VIDEO[1] + h / 2 - 150 + 60 * math.sin(2 * phase)
    draw.ellipse(box((cx - 22, cy - 22, cx + 22, cy + 22)), fill=C["faint"])
    draw.text((s(VIDEO[0] + w / 2), s(VIDEO[1] + h / 2 + 40)), label, font=font(40, True), fill=C["muted"], anchor="mm")
    draw.text((s(VIDEO[0] + w / 2), s(VIDEO[1] + h / 2 + 95)), "real sign video goes here",
              font=font(26), fill=C["faint"], anchor="mm")
    draw.rectangle(box(VIDEO), outline=C["border"], width=s(2))


def draw_icon(draw, cx, cy, correct, progress):
    scale = ease_back(progress)
    if scale <= 0:
        return
    r = 30 * scale
    draw.ellipse(box((cx - r, cy - r, cx + r, cy + r)), fill=C["green"] if correct else C["red"])
    k, lw = r / 30, s(6 * max(scale, 0.2))
    if correct:
        points = [(cx - 13 * k, cy + 1 * k), (cx - 4 * k, cy + 10 * k), (cx + 14 * k, cy - 10 * k)]
        draw.line([(s(x), s(y)) for x, y in points], fill=(255, 255, 255), width=lw, joint="curve")
    else:
        for (x0, y0), (x1, y1) in (((-11, -11), (11, 11)), ((-11, 11), (11, -11))):
            draw.line([(s(cx + x0 * k), s(cy + y0 * k)), (s(cx + x1 * k), s(cy + y1 * k))], fill=(255, 255, 255), width=lw)


def draw_truth(draw, truth, t):
    p = ease_out((t - TRUTH_AT) / TRUTH_ANIM)
    draw.rounded_rectangle(box(TRUTH), radius=s(18), fill=C["card"], outline=C["border"], width=s(2))
    draw.text((s(TRUTH[0] + 28), s(TRUTH[1] + 32)), "Ground truth", font=font(24), fill=C["muted"], anchor="lm")
    y = TRUTH[1] + 86
    if p <= 0:
        draw.text((s(TRUTH[0] + 28), s(y)), "?", font=font(52, True), fill=C["faint"], anchor="lm")
        return
    f = fitted_font(draw, truth, TRUTH[2] - TRUTH[0] - 56, 52)
    draw.text((s(TRUTH[0] + 28), s(y + 14 * (1 - p))), truth, font=f, fill=mix(C["card"], C["text"], p), anchor="lm")
    width = draw.textlength(truth, font=f) / SS
    draw.rectangle(box((TRUTH[0] + 28, y + 34, TRUTH[0] + 28 + width * p, y + 38)), fill=C["accent"])


def draw_row(draw, i, method, prediction, correct, t):
    y0 = ROW_TOP + i * (ROW_H + ROW_GAP)
    pred_p = clamp01((t - (PRED_START + i * PRED_STEP)) / PRED_ANIM)
    mark_p = clamp01((t - (MARK_START + i * MARK_STEP)) / MARK_ANIM)
    tint = ease_out(mark_p)
    good, good_bg, good_text = ((C["green"], C["green_bg"], C["green_text"]) if correct
                                else (C["red"], C["red_bg"], C["red_text"]))
    fill = mix(C["card"], good_bg, tint)
    draw.rounded_rectangle(box((PANEL_X0, y0, PANEL_X1, y0 + ROW_H)), radius=s(18), fill=fill,
                           outline=mix(C["border"], good, tint), width=s(2 + tint))
    draw.text((s(PANEL_X0 + 28), s(y0 + 44)), method["name"], font=font(32, True), fill=C["text"], anchor="lm")
    draw.text((s(PANEL_X0 + 28), s(y0 + 86)), method.get("note", ""), font=font(21), fill=C["muted"], anchor="lm")
    right = PANEL_X1 - 112
    if pred_p <= 0:
        for k in range(3):
            glow = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 1.6 - k * 0.18))
            cx = right - 60 + k * 26
            draw.ellipse(box((cx - 6, y0 + ROW_H / 2 - 6, cx + 6, y0 + ROW_H / 2 + 6)),
                         fill=mix(C["faint"], C["muted"], glow))
        return
    e = ease_out(pred_p)
    color = mix(fill, C["text"], e) if mark_p <= 0 else mix(C["text"], good_text, tint)
    f = fitted_font(draw, prediction, right - PRED_X0 - 8, 38)
    draw.text((s(right + 40 * (1 - e)), s(y0 + ROW_H / 2)), prediction, font=f, fill=color, anchor="rm")
    draw_icon(draw, PANEL_X1 - 56, y0 + ROW_H / 2, correct, mark_p)


def draw_progress(draw, fraction):
    draw.rectangle(box((VIDEO[0], PROGRESS_Y, VIDEO[0] + (VIDEO[2] - VIDEO[0]) * clamp01(fraction), PROGRESS_Y + 8)),
                   fill=C["accent"])


# ---------------------------------------------------------------- clips, audio

def load_clip(clip):
    """Decode a clip to RGB frames fitted (letterboxed) into the video area."""
    w, h = VIDEO[2] - VIDEO[0], VIDEO[3] - VIDEO[1]
    command = ["ffmpeg", "-v", "error"]
    if clip.get("trim"):
        command += ["-ss", str(clip["trim"][0]), "-to", str(clip["trim"][1])]
    command += ["-i", str(clip["video"]), "-an", "-vf",
                f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if not len(frames):
        raise ValueError(f"No frames decoded from {clip['video']}")
    return frames


def chime():
    t = np.arange(int(0.55 * SR)) / SR
    out = np.zeros_like(t)
    for start, freq, decay, level in ((0.0, 880.0, 28, 0.7), (0.07, 1318.5, 9, 1.0)):   # A5 then E6
        u = t - start
        on = u >= 0
        v = u[on]
        env = np.minimum(1, v / 0.008) * np.exp(-v * decay)
        out[on] += level * env * (np.sin(2 * np.pi * freq * v) + 0.08 * np.sin(2 * np.pi * 2 * freq * v) * np.exp(-v * 14))
    return 0.22 * out


def buzzer():
    out = []
    for freq, length in ((233.1, 0.13), (174.6, 0.27)):                  # two falling tones
        t = np.arange(int(length * SR)) / SR
        phase = 2 * np.pi * freq * t
        tone = sum(np.sin(k * phase) / k ** 2 for k in range(1, 4))      # rounded, not a harsh saw
        env = np.minimum(1, t / 0.015) * np.minimum(1, (length - t) / 0.07)
        out += [0.17 * tone * env, np.zeros(int(0.02 * SR))]
    return np.concatenate(out)


def write_audio(path, events, seconds):
    track = np.zeros(int(math.ceil(seconds * SR)) + SR)
    for at, kind in events:
        sound = chime() if kind == "correct" else buzzer()
        start = int(at * SR)
        track[start:start + len(sound)] += sound[:len(track) - start]
    track = track[:int(round(seconds * SR))]
    peak = np.abs(track).max()
    if peak > 0.5:                      # keep effects well below full scale
        track *= 0.5 / peak
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SR)
        stream.writeframes((track * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- end card

def bar_center(i):
    return BAR_TOP + i * BAR_STEP


def summary_layer(spec, methods):
    """Static parts of the end card: card, title, axis, gridlines, row labels, footnotes."""
    image = static_layer(spec, False)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box((CARD[0], CARD[1] + 6, CARD[2], CARD[3] + 6)), radius=s(22), fill=40)
    image.paste(Image.new("RGB", image.size, C["shadow"]), (0, 0), mask.filter(ImageFilter.GaussianBlur(s(10))))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box(CARD), radius=s(22), fill=C["card"], outline=C["border"], width=s(2))
    summary, left = spec["summary"], CARD[0] + 60
    draw.text((s(left), s(CARD[1] + 72)), summary["title"], font=font(40, True), fill=C["text"], anchor="lm")
    draw.text((s(left), s(CARD[1] + 120)), summary.get("subtitle", ""), font=font(24), fill=C["muted"], anchor="lm")
    top, bottom = bar_center(0) - 44, bar_center(len(methods) - 1) + 44
    for pct in (0, 25, 50, 75, 100):            # solid hairlines, baseline one step darker
        x = CHART_X0 + (CHART_X1 - CHART_X0) * pct / 100
        draw.line([(s(x), s(top)), (s(x), s(bottom))], fill=C["faint"] if pct == 0 else C["border"], width=s(1))
        draw.text((s(x), s(bottom + 24)), f"{pct}%", font=font(20), fill=C["muted"], anchor="mm")
    for i, method in enumerate(methods):
        y = bar_center(i)
        draw.text((s(left), s(y - 13)), method["name"], font=font(30, True), fill=C["text"], anchor="lm")
        if method.get("note"):
            draw.text((s(left), s(y + 20)), method["note"], font=font(20), fill=C["muted"], anchor="lm")
    for k, line in enumerate(summary.get("footnotes", [])):
        f = fitted_font(draw, line, CARD[2] - CARD[0] - 120, 21, bold=False, minimum=16)
        draw.text((s(left), s(CARD[3] - 92 + 34 * k)), line, font=f, fill=C["muted"], anchor="lm")
    return image


def draw_bars(draw, values, t):
    """Bars grow from the baseline; 4px rounded data end, square at the baseline; value at the tip."""
    for i, value in enumerate(values):
        p = ease_out((t - 0.35 - i * 0.12) / 0.75)
        if p <= 0:
            continue
        y = bar_center(i)
        x1 = CHART_X0 + (CHART_X1 - CHART_X0) * value / 100 * p
        if x1 - CHART_X0 >= 8:
            draw.rounded_rectangle(box((CHART_X0, y - BAR_HALF, x1, y + BAR_HALF)), radius=s(4), fill=BAR)
            draw.rectangle(box((CHART_X0, y - BAR_HALF, CHART_X0 + 4, y + BAR_HALF)), fill=BAR)
        elif x1 > CHART_X0:
            draw.rectangle(box((CHART_X0, y - BAR_HALF, x1, y + BAR_HALF)), fill=BAR)
        shown = clamp01((p - 0.6) / 0.4)
        if shown > 0:                            # text wears text ink, never the bar color
            draw.text((s(x1 + 14), s(y)), f"{value:.2f}%", font=font(28, True), fill=mix(C["card"], C["text"], shown), anchor="lm")


# ---------------------------------------------------------------- main render

def validate(spec):
    methods, clips = spec.get("methods"), spec.get("clips")
    if not isinstance(methods, list) or len(methods) != 4:
        raise ValueError("Exactly four methods are required for this layout")
    if not clips:
        raise ValueError("At least one clip is required")
    for n, clip in enumerate(clips, 1):
        if len(clip.get("predictions", [])) != len(methods) or not clip.get("truth"):
            raise ValueError(f"Clip {n}: need a truth label and one prediction per method")
        if clip.get("video") and not Path(clip["video"]).is_file():
            raise ValueError(f"Clip {n}: video not found: {clip['video']}")
    summary = spec.get("summary")
    if summary is not None:
        values = summary.get("values", [])
        if not summary.get("title") or len(values) != len(methods) or not all(0 <= v <= 100 for v in values):
            raise ValueError("Summary needs a title and one percentage (0-100) per method, in method order")


def render(spec, output):
    validate(spec)
    clips, methods = spec["clips"], spec["methods"]
    title_frames, segment_frames = round(TITLE_S * FPS), round(SEGMENT_S * FPS)
    summary_frames = round(SUMMARY_S * FPS) if spec.get("summary") else 0
    total = title_frames + len(clips) * segment_frames + summary_frames
    main_static, title_static = static_layer(spec, True), static_layer(spec, False)
    main_static_1x, title_static_1x = main_static.reduce(SS), title_static.reduce(SS)

    events = []
    for n, clip in enumerate(clips):
        base = TITLE_S + n * SEGMENT_S
        for i, prediction in enumerate(clip["predictions"]):
            events.append((base + MARK_START + i * MARK_STEP, "correct" if prediction == clip["truth"] else "wrong"))

    with tempfile.TemporaryDirectory() as folder:
        audio = Path(folder) / "audio.wav"
        write_audio(audio, events, total / FPS)
        encoder = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(audio), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)],
            stdin=subprocess.PIPE)

        def emit(image):
            encoder.stdin.write(image.tobytes())

        for f in range(title_frames):
            t = f / FPS
            image = title_static.copy()
            draw = ImageDraw.Draw(image)
            p = ease_out(t / 0.5)
            lift = 20 * (1 - p)
            draw.text((s(W / 2), s(450 + lift)), spec["title"], font=font(72, True), fill=mix(C["bg"], C["text"], p), anchor="mm")
            draw.text((s(W / 2), s(545 + lift)), " · ".join(m["name"] for m in methods), font=font(34),
                      fill=mix(C["bg"], C["muted"], p), anchor="mm")
            if spec.get("course"):
                draw.text((s(W / 2), s(615 + lift)), spec["course"], font=font(26), fill=mix(C["bg"], C["faint"], p), anchor="mm")
            frame = image.reduce(SS)
            fade = ease_out((TITLE_S - t) / FADE_S)
            emit(Image.blend(title_static_1x, frame, fade) if fade < 1 else frame)

        for n, clip in enumerate(clips):
            frames = load_clip(clip) if clip.get("video") else None
            loop_s = len(frames) / FPS if frames is not None else PLACEHOLDER_LOOP_S
            correct = [prediction == clip["truth"] for prediction in clip["predictions"]]
            for f in range(segment_frames):
                t = f / FPS
                image = main_static.copy()
                draw = ImageDraw.Draw(image)
                draw.text((s(1830), s(78)), f"Clip {n + 1} / {len(clips)}", font=font(30), fill=C["muted"], anchor="rm")
                if frames is None:
                    draw_placeholder(image, clip.get("placeholder", f"Clip {n + 1}"), t)
                draw_progress(draw, (t % loop_s) / loop_s)
                draw_truth(draw, clip["truth"], t)
                for i, method in enumerate(methods):
                    draw_row(draw, i, method, clip["predictions"][i], correct[i], t)
                frame = image.reduce(SS)
                if frames is not None:
                    frame.paste(Image.fromarray(frames[f % len(frames)]), (VIDEO[0], VIDEO[1]))
                fade = min(ease_out(t / FADE_S), ease_out((SEGMENT_S - t) / FADE_S))
                emit(Image.blend(main_static_1x, frame, fade) if fade < 1 else frame)

        if summary_frames:
            card = summary_layer(spec, methods)
            for f in range(summary_frames):
                t = f / FPS
                image = card.copy()
                draw_bars(ImageDraw.Draw(image), spec["summary"]["values"], t)
                frame = image.reduce(SS)
                fade = ease_out(t / 0.3)          # fade in, then hold on the last frame
                emit(Image.blend(title_static_1x, frame, fade) if fade < 1 else frame)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    return total / FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for clip in spec.get("clips", []):          # relative video paths are relative to the spec file
        if clip.get("video") and not Path(clip["video"]).is_absolute():
            clip["video"] = str((args.spec.parent / clip["video"]).resolve())
    seconds = render(spec, args.output)
    print(f"Wrote {args.output} ({seconds:.1f} s)")


if __name__ == "__main__":
    main()
