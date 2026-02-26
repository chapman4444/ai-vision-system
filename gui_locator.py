#!/usr/bin/env python3
"""
gui_locator.py — generalized GUI element detector (OCR + geometry + color), CLI.

Purpose:
  Detects generic GUI primitives in a screenshot (apps or webpages), including:
    - text runs (OCR lines, words)
    - rectangle-like widgets: buttons, text inputs, text areas, menu bars
    - dropdown/combo boxes (rectangle + triangle/chevron glyph)
    - checkboxes (squares) and radio buttons (circles)
    - generic rectangles (fallback)
  Outputs JSON describing each element (type, bbox, center, text, score, features).
  Can save a debug overlay. Optionally triggers a click via your existing clicker.

Usage:
  python gui_locator.py --image "E:\\Devel\\viewer\\llm_screenshots\\current_view.png" \
                        --config config.json --save-overlay \
                        --out detections.json
  # Click the Nth detection (after filtering) using llm_click.py
  python gui_locator.py --image ... --config config.json --filter type=button --click nth=0

Dependencies:
  pip install opencv-python pillow numpy pytesseract
  (Install Tesseract and, if needed, set its path in config.json: "tesseract_cmd": "...")

Notes:
  - Designed for fast-changing images: robust loader retries while file is being written.
  - No keyword lock-in: config provides optional weights/thresholds and color hints.
  - Elements are *scored* by multiple signals; higher = more confident.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("[WARN] pytesseract not installed; OCR features disabled")


# ----------------------------- data model -----------------------------

@dataclass
class Element:
    id: int
    type: str                    # button | text_input | text_area | dropdown | combo_box | checkbox | radio | menu_bar | text | rect
    bbox: Tuple[int, int, int, int]  # x, y, w, h (image coords)
    center: Tuple[int, int]
    text: str
    score: float
    features: Dict[str, Any]


# ----------------------------- utils -----------------------------

def load_bgr_stable(path: str, retries: int = 4, pause_ms: int = 60) -> np.ndarray:
    """
    Robust loader for fast-changing files: read twice until identical.
    """
    path = str(path)
    last = None
    for _ in range(retries):
        a = np.fromfile(path, dtype=np.uint8)
        time.sleep(pause_ms / 1000.0)
        b = np.fromfile(path, dtype=np.uint8)
        if a.size and a.size == b.size and np.array_equal(a, b):
            last = b
            break
    if last is None:
        last = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(last, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def save_image_any(path: str, bgr: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    cv2.imencode(ext, bgr)[1].tofile(path)


def clamp_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def center_of(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return int(x + w / 2), int(y + h / 2)


def avg_intensity(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    roi = gray[y:y+h, x:x+w]
    return float(np.mean(roi)) if roi.size else 0.0


def aspect(w: int, h: int) -> float:
    return (w / float(h)) if h else 0.0


def rectness(contour: np.ndarray, w: int, h: int) -> float:
    # how rectangle-like: contour area / (w*h)
    area = cv2.contourArea(contour)
    if w * h == 0:
        return 0.0
    return float(area) / float(w * h)


def to_bgr(rgb: List[int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


# ----------------------------- OCR -----------------------------

def ocr_lines(gray: np.ndarray, config: Dict[str, Any]) -> List[Element]:
    """
    Extract text runs (grouped as lines) using Tesseract.
    """
    if pytesseract is None:
        return []
    psm = int(config.get("ocr_psm", 6))
    conf_thresh = int(config.get("ocr_confidence", 55))
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=f"--oem 3 --psm {psm}")

    # group by (page, block, par, line)
    groups: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]] = {}
    n = len(data["text"])
    for i in range(n):
        try:
            conf = int(float(data["conf"][i]))
        except ValueError:
            conf = -1
        if conf < conf_thresh:
            continue
        txt = data["text"][i].strip()
        if not txt:
            continue
        key = (
            int(data["page_num"][i]),
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        item = dict(
            text=txt,
            left=int(data["left"][i]),
            top=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
            word_num=int(data["word_num"][i]),
            conf=conf,
        )
        groups.setdefault(key, []).append(item)

    # produce Elements
    elems: List[Element] = []
    eid = 0
    for _, words in groups.items():
        words = sorted(words, key=lambda d: d["word_num"])
        xs = [w["left"] for w in words]
        ys = [w["top"] for w in words]
        xe = [w["left"] + w["width"] for w in words]
        ye = [w["top"] + w["height"] for w in words]
        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
        text = " ".join([w["text"] for w in words])
        elems.append(Element(
            id=eid, type="text", bbox=(x1, y1, x2 - x1, y2 - y1), center=center_of(x1, y1, x2 - x1, y2 - y1),
            text=text, score=float(np.mean([w["conf"] for w in words])), features={"words": len(words)}
        ))
        eid += 1
    return elems


# ----------------------------- geometry detectors -----------------------------

def detect_shapes(bgr: np.ndarray, gray: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Return (contour, bbox) for reasonably-rectangular or roundish shapes.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, cfg.get("canny_low", 50), cfg.get("canny_high", 120))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    H, W = gray.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # generic limits
        if w * h < cfg.get("min_area_px", 150) or w < 6 or h < 6:
            continue
        if w > W * 0.98 and h > H * 0.98:
            continue
        out.append((c, (x, y, w, h)))
    return out


def find_triangles(bin_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find triangle-like contours (for dropdown arrows).
    Returns bboxes.
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tri = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(approx)
            tri.append((x, y, w, h))
    return tri


# ----------------------------- classifier -----------------------------

def classify_elements(bgr: np.ndarray, gray: np.ndarray, shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
                      texts: List[Element], cfg: Dict[str, Any]) -> List[Element]:
    H, W = gray.shape[:2]
    out: List[Element] = []
    eid = 0

    # Precompute text index for quick overlaps
    text_boxes = [(t, t.bbox) for t in texts]

    for c, (x, y, w, h) in shapes:
        cx, cy = center_of(x, y, w, h)
        rness = rectness(c, w, h)
        ar = aspect(w, h)
        mean = avg_intensity(gray, x, y, w, h)

        # Basic features
        feats: Dict[str, Any] = {"rectness": round(rness, 3), "aspect": round(ar, 3), "mean": round(mean, 1), "w": w, "h": h}

        # Overlap with any OCR line
        overlap_text = ""
        overlap_count = 0
        for t, (tx, ty, tw, th) in text_boxes:
            if not (x > tx + tw or tx > x + w or y > ty + th or ty > y + h):
                overlap_count += 1
                overlap_text = (overlap_text + " " + t.text).strip()
        feats["overlap_text_count"] = overlap_count

        # Dropdown glyph check: look at rightmost third for a triangle
        tri_hit = False
        if cfg.get("detect_dropdown_glyph", True) and w >= 24 and h >= 14:
            rx1 = x + int(w * 0.65)
            rx2 = x + w
            ry1 = y + int(h * 0.15)
            ry2 = y + int(h * 0.85)
            rx1, ry1, rw, rh = clamp_bbox(rx1, ry1, rx2 - rx1, ry2 - ry1, W, H)
            if rw > 4 and rh > 4:
                sub = gray[ry1:ry1+rh, rx1:rx1+rw]
                sub = cv2.Canny(sub, 60, 150)
                tris = find_triangles(sub)
                tri_hit = len(tris) > 0
        feats["triangle_glyph"] = tri_hit

        # Checkbox / radio by size + roundness
        area = w * h
        small_sq = min(w, h) >= 10 and max(w, h) <= 36 and 0.75 <= ar <= 1.33 and rness > 0.5
        circ_like = False
        if small_sq is False and max(w, h) <= 40:
            # circularity ~ 4πA / P^2; use approx
            A = cv2.contourArea(c)
            P = cv2.arcLength(c, True)
            circ = (4 * np.pi * A / (P * P + 1e-6)) if P > 0 else 0.0
            circ_like = circ > 0.65
            feats["circularity"] = round(circ, 3)

        # Candidate type scores (all weak-learners)
        score_rect = rness
        score_button = (rness > 0.6) * 0.4 + (1.3 <= ar <= 6.0) * 0.3 + (14 <= h <= 60) * 0.2 + (overlap_count > 0) * 0.2
        score_text_input = (rness > 0.6) * 0.3 + (ar >= 2.0) * 0.3 + (18 <= h <= 40) * 0.2 + (mean > 170) * 0.2
        score_text_area = (rness > 0.6) * 0.3 + (ar >= 1.2) * 0.2 + (h >= 70) * 0.3 + (mean > 170) * 0.2
        score_menu_bar = (y < H * 0.2) * 0.3 + (ar >= 4.0) * 0.3 + (h <= 70) * 0.2 + (overlap_count >= 2) * 0.2
        score_dropdown = (score_text_input > 0.4) * 0.2 + (tri_hit) * 0.7 + (overlap_count >= 0) * 0.1
        score_checkbox = small_sq * 1.0
        score_radio = circ_like * 1.0

        # Choose best label by score
        scores = {
            "button": float(score_button),
            "text_input": float(score_text_input),
            "text_area": float(score_text_area),
            "menu_bar": float(score_menu_bar),
            "dropdown": float(score_dropdown),
            "checkbox": float(score_checkbox),
            "radio": float(score_radio),
            "rect": float(score_rect),
        }
        label = max(scores.items(), key=lambda kv: kv[1])[0]
        score = float(scores[label])

        # Compose element
        out.append(Element(
            id=eid, type=label, bbox=(x, y, w, h), center=(cx, cy),
            text=overlap_text, score=round(score, 3), features=feats
        ))
        eid += 1

    # Include raw OCR lines as separate elements (already constructed)
    # (They’re useful targets for LLMs even without a box.)
    for t in texts:
        eid += 1
        out.append(t)

    return out


# ----------------------------- color rules (optional) -----------------------------

def apply_color_hints(bgr: np.ndarray, elems: List[Element], cfg: Dict[str, Any]) -> None:
    """
    Optionally up/down-weight elements if their mean color fits configured hints.
    config["color_hints"] = { "button": {"rgb":[200,200,200], "tol":30}, ... }
    """
    hints = cfg.get("color_hints", {})
    if not hints:
        return
    for e in elems:
        hint = hints.get(e.type)
        if not hint:
            continue
        rgb = hint["rgb"]
        tol = int(hint.get("tol", 25))
        x, y, w, h = e.bbox
        roi = bgr[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
        target = np.array(to_bgr(rgb), dtype=np.float32)
        if np.all(np.abs(mean_bgr - target) <= tol):
            e.score = round(min(1.0, e.score + 0.15), 3)
        else:
            e.score = round(max(0.0, e.score - 0.05), 3)


# ----------------------------- overlay -----------------------------

def draw_overlay(bgr: np.ndarray, elems: List[Element], cfg: Dict[str, Any]) -> np.ndarray:
    out = bgr.copy()
    for e in elems:
        x, y, w, h = e.bbox
        color = (0, 200, 0)
        if e.type in ("checkbox", "radio"): color = (0, 180, 255)
        if e.type in ("menu_bar",): color = (255, 160, 0)
        if e.type in ("text",): color = (200, 200, 200)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cx, cy = e.center
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        label = f"{e.type}:{e.score:.2f}"
        if e.text:
            short = (e.text[:24] + "…") if len(e.text) > 24 else e.text
            label += f" [{short}]"
        cv2.putText(out, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ----------------------------- filtering / sorting -----------------------------

def filter_elements(elems: List[Element], filters: List[str]) -> List[Element]:
    """
    filters like: ["type=button", "text~=center", "score>=0.6", "w>40"]
    ops: =,~=,>=,>,<=,<  ; text~= does case-insensitive substring on text/type
    """
    def match(e: Element) -> bool:
        local = {"w": e.bbox[2], "h": e.bbox[3], "score": e.score}
        for f in filters:
            if "~=" in f:
                key, val = f.split("~=", 1)
                key = key.strip()
                val = val.strip().lower()
                field = getattr(e, key, "") if hasattr(e, key) else (e.text if key == "text" else e.type)
                if val not in str(field).lower():
                    return False
            elif "=" in f and "==" not in f and ">=" not in f and "<=" not in f:
                key, val = f.split("=", 1)
                key = key.strip()
                val = val.strip().lower()
                if key == "type" and e.type.lower() != val:
                    return False
                elif key == "text" and val not in e.text.lower():
                    return False
            else:
                for op in (">=", "<=", ">", "<"):
                    if op in f:
                        key, val = f.split(op, 1)
                        a = float(local.get(key.strip(), -1e9))
                        b = float(val.strip())
                        if op == ">=" and not (a >= b): return False
                        if op == "<=" and not (a <= b): return False
                        if op == ">" and not (a > b): return False
                        if op == "<" and not (a < b): return False
                        break
        return True

    return [e for e in elems if match(e)]


# ----------------------------- main pipeline -----------------------------

def run_detection(image_path: str, cfg: Dict[str, Any]) -> List[Element]:
    if cfg.get("tesseract_cmd"):
        pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]

    bgr = load_bgr_stable(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 15, 15)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # OCR lines
    text_elems = ocr_lines(gray, cfg)

    # Shapes
    shapes = detect_shapes(bgr, gray, cfg)

    # Classify
    elems = classify_elements(bgr, gray, shapes, text_elems, cfg)

    # Optional color hints
    apply_color_hints(bgr, elems, cfg)

    # Sort by score desc
    elems.sort(key=lambda e: e.score, reverse=True)
    return elems


def maybe_click(elems: List[Element], cfg: Dict[str, Any], click_arg: str) -> None:
    """
    click_arg: "nth=0" or "id=12"
    Uses llm_click.py by default; or viewer_click.py if click_mode="viewer"
    """
    target: Optional[Element] = None
    if click_arg.startswith("nth="):
        idx = int(click_arg.split("=", 1)[1])
        if 0 <= idx < len(elems):
            target = elems[idx]
    elif click_arg.startswith("id="):
        eid = int(click_arg.split("=", 1)[1])
        for e in elems:
            if e.id == eid:
                target = e
                break
    if not target:
        print(json.dumps({"status": "error", "message": "click target not found"}))
        return

    x, y = target.center
    mode = cfg.get("click_mode", "llm")  # "llm" | "viewer"
    if mode == "viewer":
        exe = [sys.executable, "viewer_click.py", str(x), str(y)]
    else:
        exe = [sys.executable, "llm_click.py", str(x), str(y)]

    try:
        print(f"# click: {mode} @ ({x},{y})")
        subprocess.run(exe, check=False)
    except Exception as e:
        print(f"# click error: {e}")


def main():
    ap = argparse.ArgumentParser(description="General GUI detector (OCR + geometry + color)")
    ap.add_argument("--image", default=r"E:\Devel\viewer\llm_screenshots\current_view.png")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--filter", action="append", default=[], help="e.g. type=button ; score>=0.6 ; text~=ok")
    ap.add_argument("--click", default="", help="nth=0 or id=12 (uses llm_click.py by default)")
    args = ap.parse_args()

    # load config
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

    # detect
    elems = run_detection(args.image, cfg)

    # filter if requested
    if args.filter:
        elems = filter_elements(elems, args.filter)

    # dump JSON
    payload = {"image": args.image, "count": len(elems), "elements": [asdict(e) for e in elems]}
    print(json.dumps(payload, indent=2))

    # save overlay
    if args.save_overlay:
        bgr = load_bgr_stable(args.image)
        overlay = draw_overlay(bgr, elems, cfg)
        out_path = str(Path(args.image).with_name(Path(args.image).stem + "_overlay.png"))
        save_image_any(out_path, overlay)

    # write out file
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # optional click
    if args.click:
        maybe_click(elems, cfg, args.click)


if __name__ == "__main__":
    main()
