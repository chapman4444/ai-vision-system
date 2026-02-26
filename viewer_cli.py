"""CLI for LLM-driven GUI automation: capture, click, move, type, OCR, and color search."""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

# Optional cross-platform helpers
try:
    import mss  # cross-platform screen capture
except Exception:  # pragma: no cover
    mss = None

try:
    import pytesseract  # OCR
except Exception:  # pragma: no cover
    pytesseract = None

# ------------------------------------------------------------
# Platform feature detection (fixes AttributeError: ctypes.windll)
# ------------------------------------------------------------
IS_WINDOWS = (os.name == "nt") and hasattr(ctypes, "windll")
if IS_WINDOWS:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
else:
    # Placeholders so module import works in non-Windows / sandboxed envs
    user32 = None
    gdi32 = None

# Virtual screen metrics constants (Windows)
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_CMONITORS = 80

# Mouse event flags (Windows)
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

# Keyboard input flags (Windows keybd_event)
KEYEVENTF_KEYUP = 0x0002

# Simulated cursor position for non-Windows fallback
_SIM_CURSOR = [100, 100]


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


# ------------------------------------------------------------
# Monitor & virtual screen helpers (Windows + cross-platform fallbacks)
# ------------------------------------------------------------

def get_virtual_screen_rect():
    """Return the virtual desktop rect as (x, y, w, h).
    On non-Windows, use mss if available; otherwise return a sane default."""
    if IS_WINDOWS:
        x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        return x, y, w, h
    # Cross-platform path
    if mss is not None:
        with mss.mss() as sct:
            mon = sct.monitors[0]  # virtual bounding box over all monitors
            return mon["left"], mon["top"], mon["width"], mon["height"]
    # Fallback default
    return 0, 0, 1280, 720


def enum_monitors():
    """Enumerate monitors; return list of dicts with 'left','top','right','bottom','primary','device'.
    Windows uses EnumDisplayMonitors. Else, uses mss if available, or a single virtual monitor."""
    if IS_WINDOWS:
        MONITORINFOF_PRIMARY = 0x00000001
        monitors = []
        MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, wt.HMONITOR, wt.HDC, wt.LPRECT, ctypes.c_double)

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wt.DWORD),
                ("rcMonitor", wt.RECT),
                ("rcWork", wt.RECT),
                ("dwFlags", wt.DWORD),
                ("szDevice", wt.WCHAR * 32),
            ]

        def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            info = MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(MONITORINFOEXW)
            user32.GetMonitorInfoW(hMonitor, ctypes.byref(info))
            rect = info.rcMonitor
            monitors.append({
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
                "primary": bool(info.dwFlags & MONITORINFOF_PRIMARY),
                "device": info.szDevice,
            })
            return 1

        cb = MONITORENUMPROC(_callback)
        user32.EnumDisplayMonitors(0, 0, cb, 0)
        return monitors

    # Cross-platform using mss
    if mss is not None:
        out = []
        with mss.mss() as sct:
            mons = sct.monitors[1:]  # 1..N
            for i, mon in enumerate(mons, start=1):
                out.append({
                    "left": mon["left"],
                    "top": mon["top"],
                    "right": mon["left"] + mon["width"],
                    "bottom": mon["top"] + mon["height"],
                    "primary": (i == 1),
                    "device": f"MSS{i}",
                })
        return out

    # Fallback single virtual monitor
    x, y, w, h = get_virtual_screen_rect()
    return [{"left": x, "top": y, "right": x + w, "bottom": y + h, "primary": True, "device": "VIRTUAL"}]


def world_from_monitor(monitor_index: int, x_local: int, y_local: int):
    """Convert monitor-local (x_local,y_local) to world coords (wx, wy).
    monitor_index is 1-based."""
    mons = enum_monitors()
    if monitor_index < 1 or monitor_index > len(mons):
        raise ValueError(f"monitor_index out of range: {monitor_index}")
    m = mons[monitor_index - 1]
    wx = m["left"] + int(x_local)
    wy = m["top"] + int(y_local)
    return wx, wy


# ------------------------------------------------------------
# Cursor helpers (real on Windows; simulated elsewhere)
# ------------------------------------------------------------

def get_cursor_pos():
    if IS_WINDOWS:
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    # Simulated fallback
    return int(_SIM_CURSOR[0]), int(_SIM_CURSOR[1])


def set_cursor_pos_world(wx: int, wy: int):
    if IS_WINDOWS:
        if not user32.SetCursorPos(int(wx), int(wy)):
            raise OSError("SetCursorPos failed")
        return
    # Simulated: update internal cursor only
    _SIM_CURSOR[0], _SIM_CURSOR[1] = int(wx), int(wy)


# ------------------------------------------------------------
# Move & click (clicks are no-ops off Windows but reported clearly)
# ------------------------------------------------------------

def move_mouse_monitor(monitor_index: int, x_local: int, y_local: int, verify_radius: int = 2, recap_out: Path | None = None):
    wx, wy = world_from_monitor(monitor_index, x_local, y_local)
    set_cursor_pos_world(wx, wy)
    time.sleep(0.01)
    cx, cy = get_cursor_pos()
    ok = abs(cx - wx) <= verify_radius and abs(cy - wy) <= verify_radius

    payload = {
        "event": "mouse_move",
        "monitor": monitor_index,
        "x": int(x_local),
        "y": int(y_local),
        "world": {"x": wx, "y": wy},
        "cursor": {"x": cx, "y": cy},
        "ok": bool(ok),
        "simulated": (not IS_WINDOWS),
    }

    if recap_out:
        try:
            capture_current_view(Path(recap_out), include_cursor=True)
            payload["recapture"] = str(Path(recap_out).resolve())
        except Exception as e:
            payload["recapture_error"] = str(e)

    print(json.dumps(payload))
    return ok


def click_mouse_monitor(
    monitor_index: int,
    x_local: int,
    y_local: int,
    button: str = "left",
    double: bool = False,
    verify_radius: int = 2,
    recap_out: Path | None = None,
):
    ok_move = move_mouse_monitor(monitor_index, x_local, y_local, verify_radius=verify_radius)

    btn_down = btn_up = 0
    if button == "left":
        btn_down, btn_up = MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
    elif button == "right":
        btn_down, btn_up = MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    elif button == "middle":
        btn_down, btn_up = MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
    else:
        raise ValueError("button must be left|right|middle")

    performed = False
    if IS_WINDOWS:
        def _click_once():
            user32.mouse_event(btn_down, 0, 0, 0, 0)
            user32.mouse_event(btn_up, 0, 0, 0, 0)
        _click_once()
        if double:
            time.sleep(0.05)
            _click_once()
        performed = True

    cx, cy = get_cursor_pos()
    payload = {
        "event": "mouse_click",
        "monitor": monitor_index,
        "x": int(x_local),
        "y": int(y_local),
        "button": button,
        "double": bool(double),
        "cursor": {"x": cx, "y": cy},
        "ok": bool(ok_move and performed),
        "performed": performed,
        "simulated": (not IS_WINDOWS),
    }

    if recap_out:
        try:
            capture_current_view(Path(recap_out), include_cursor=True)
            payload["recapture"] = str(Path(recap_out).resolve())
        except Exception as e:
            payload["recapture_error"] = str(e)

    print(json.dumps(payload))
    return ok_move and performed


# ------------------------------------------------------------
# Image capture (Windows GDI, else MSS if available, else placeholder)
# ------------------------------------------------------------

def overlay_cursor(img: Image.Image, pos, cursor_type: str = "cross", cursor_size: int = 15):
    if pos is None:
        return img
    x, y = pos
    draw = ImageDraw.Draw(img)
    if cursor_type == "cross":
        sz = max(3, int(cursor_size))
        draw.line((x - sz, y, x + sz, y))
        draw.line((x, y - sz, x, y + sz))
    elif cursor_type == "dot":
        r = max(2, int(cursor_size // 3))
        draw.ellipse((x - r, y - r, x + r, y + r))
    return img


def _capture_windows(vx: int, vy: int, vw: int, vh: int) -> Image.Image:
    # Create DCs
    hdc_screen = user32.GetDC(0)
    if not hdc_screen:
        raise OSError("GetDC failed")
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    if not hdc_mem:
        user32.ReleaseDC(0, hdc_screen)
        raise OSError("CreateCompatibleDC failed")

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wt.DWORD),
            ("biWidth", wt.LONG),
            ("biHeight", wt.LONG),
            ("biPlanes", wt.WORD),
            ("biBitCount", wt.WORD),
            ("biCompression", wt.DWORD),
            ("biSizeImage", wt.DWORD),
            ("biXPelsPerMeter", wt.LONG),
            ("biYPelsPerMeter", wt.LONG),
            ("biClrUsed", wt.DWORD),
            ("biClrImportant", wt.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]

    bi = BITMAPINFO()
    bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.bmiHeader.biWidth = vw
    bi.bmiHeader.biHeight = -vh  # top-down DIB
    bi.bmiHeader.biPlanes = 1
    bi.bmiHeader.biBitCount = 32
    bi.bmiHeader.biCompression = 0  # BI_RGB

    ppv_bits = ctypes.c_void_p()
    hbm = gdi32.CreateDIBSection(hdc_screen, ctypes.byref(bi), 0, ctypes.byref(ppv_bits), 0, 0)
    if not hbm:
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        raise OSError("CreateDIBSection failed")

    gdi32.SelectObject(hdc_mem, hbm)

    # SRCCOPY from the virtual desktop origin (vx,vy)
    if not gdi32.BitBlt(hdc_mem, 0, 0, vw, vh, hdc_screen, vx, vy, 0x00CC0020):  # SRCCOPY
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        raise OSError("BitBlt failed")

    # Build PIL image from raw bits
    buf_len = vw * vh * 4
    buf = (ctypes.c_byte * buf_len).from_address(ppv_bits.value)
    img = Image.frombuffer("BGRA", (vw, vh), buf, "raw", "BGRA", 0, 1).convert("RGBA")

    # Cleanup
    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

    return img


def _capture_mss(vx: int, vy: int, vw: int, vh: int) -> Image.Image:
    with mss.mss() as sct:
        mon = {"left": vx, "top": vy, "width": vw, "height": vh}
        shot = sct.grab(mon)
        # Convert to PIL Image (RGB)
        img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
        return img


def capture_current_view(out_path: Path, include_cursor: bool = True, cursor_type: str = "cross", cursor_size: int = 15):
    """Capture the entire virtual desktop to out_path and optionally draw a cursor overlay.
    Works on Windows via GDI; elsewhere uses mss if available, else creates a placeholder image."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vx, vy, vw, vh = get_virtual_screen_rect()

    # Choose capture backend
    if IS_WINDOWS:
        img = _capture_windows(vx, vy, vw, vh)
    elif mss is not None:
        img = _capture_mss(vx, vy, vw, vh)
    else:
        # Placeholder image when we cannot capture the screen
        img = Image.new("RGBA", (max(1, vw), max(1, vh)), (30, 30, 30, 255))
        draw = ImageDraw.Draw(img)
        msg = "Screen capture unavailable (no Windows GDI or mss)."
        draw.text((10, 10), msg, fill=(255, 255, 255, 255))

    cursor_pos = get_cursor_pos() if include_cursor else None
    img = overlay_cursor(img, cursor_pos, cursor_type=cursor_type, cursor_size=cursor_size)
    img.save(out_path)

    print(json.dumps({
        "event": "capture",
        "file": str(out_path.resolve()),
        "virtual_rect": {"x": vx, "y": vy, "w": vw, "h": vh},
        "cursor": {"x": cursor_pos[0], "y": cursor_pos[1]} if include_cursor and cursor_pos else None,
        "simulated": (not IS_WINDOWS and mss is None),
    }))
    return str(out_path)


# ------------------------------------------------------------
# File copy utility
# ------------------------------------------------------------

def copy_current_view_to_folder(src: Path, dest_folder: Path):
    src = Path(src)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / src.name
    shutil.copy2(src, dest)
    print(json.dumps({"event": "copy_view", "src": str(src), "dest": str(dest)}))
    return str(dest)


# ------------------------------------------------------------
# OCR helpers (pytesseract)
# ------------------------------------------------------------

def ocr_screen_return_words(img_path: Path, lang: str = "eng"):
    if pytesseract is None:
        payload = {
            "event": "ocr",
            "ok": False,
            "error": "pytesseract not available; install Tesseract and pytesseract",
        }
        print(json.dumps(payload))
        return []
    img = Image.open(img_path)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    results = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append({
            "text": txt,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "center": {"x": x + w // 2, "y": y + h // 2},
        })
    payload = {"event": "ocr", "ok": True, "count": len(results), "words": results}
    print(json.dumps(payload))
    return results


# ------------------------------------------------------------
# OCR-based snapping and color-based snapping
# ------------------------------------------------------------

def snap_to_text(img_path: Path, query: str, monitor_index: int, fuzzy: bool = False, click: bool = False):
    words = ocr_screen_return_words(img_path)
    if not words:
        return False
    q = query.strip().lower()
    best = None
    best_score = -1
    for w in words:
        t = w["text"].lower()
        score = 0
        if not fuzzy:
            if q == t:
                score = 100
            elif q in t:
                score = 80
        else:
            inter = len(set(q) & set(t))
            score = int(100 * inter / max(1, len(set(q))))
            if q in t:
                score = max(score, 90)
        if score > best_score:
            best, best_score = w, score
    if best is None:
        print(json.dumps({"event": "snap_text", "ok": False, "reason": "no_match"}))
        return False
    cx, cy = best["center"]["x"], best["center"]["y"]
    vx, vy, _, _ = get_virtual_screen_rect()
    wx, wy = vx + cx, vy + cy
    set_cursor_pos_world(wx, wy)
    time.sleep(0.01)
    cpos = get_cursor_pos()
    ok_move = (abs(cpos[0] - wx) <= 2 and abs(cpos[1] - wy) <= 2) and IS_WINDOWS
    if click and ok_move and IS_WINDOWS:
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    print(json.dumps({
        "event": "snap_text",
        "ok": bool(ok_move),
        "query": query,
        "target_center": {"x": cx, "y": cy},
        "simulated": (not IS_WINDOWS),
    }))
    return ok_move


def hex_to_rgb(hex_color: str):
    s = hex_color.strip().lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    if len(s) != 6:
        raise ValueError("hex color must be #RRGGBB or #RGB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def list_buttons_by_color(img_path: Path, hex_color: str, tolerance: int = 16, direction: str = "lr", nearest: bool = False):
    """Simple color pixel finder: returns pixel positions matching color within tolerance.
    direction: lr|rl|tb|bt ordering.
    nearest: if True, order by distance from current cursor position.
    """
    rgb = hex_to_rgb(hex_color)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pixels = img.load()

    def within(a, b):
        return abs(a - b) <= tolerance

    xs = list(range(w))
    ys = list(range(h))
    if direction == "rl":
        xs = list(range(w - 1, -1, -1))
    if direction == "bt":
        ys = list(range(h - 1, -1, -1))

    matches = []
    for y in ys:
        for x in xs:
            r, g, b = pixels[x, y]
            if within(r, rgb[0]) and within(g, rgb[1]) and within(b, rgb[2]):
                matches.append({"x": x, "y": y})

    if nearest and matches:
        cx, cy = get_cursor_pos()
        vx, vy, _, _ = get_virtual_screen_rect()
        # Convert world cursor pos to virtual-image coords
        ix, iy = cx - vx, cy - vy
        matches.sort(key=lambda p: (p["x"] - ix) ** 2 + (p["y"] - iy) ** 2)

    print(json.dumps({
        "event": "list_buttons_by_color",
        "color": hex_color,
        "tolerance": tolerance,
        "direction": direction,
        "count": len(matches),
        "pixels": matches,
    }))
    return matches


def find_color_regions(img_path: Path, hex_color: str, tolerance: int = 16, direction: str = "lr"):
    """Group adjacent matching pixels into regions; return list of regions with bbox, center, and area.
    Regions are ordered by the first encountered pixel according to the scan direction."""
    rgb = hex_to_rgb(hex_color)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pixels = img.load()

    visited = [[False] * w for _ in range(h)]

    def within(a, b):
        return abs(a - b) <= tolerance

    xs = list(range(w))
    ys = list(range(h))
    if direction == "rl":
        xs = list(range(w - 1, -1, -1))
    if direction == "bt":
        ys = list(range(h - 1, -1, -1))

    regions = []
    for y in ys:
        for x in xs:
            if visited[y][x]:
                continue
            r, g, b = pixels[x, y]
            if not (within(r, rgb[0]) and within(g, rgb[1]) and within(b, rgb[2])):
                continue
            # BFS flood fill
            q = [(x, y)]
            visited[y][x] = True
            minx = maxx = x
            miny = maxy = y
            count = 0
            while q:
                px, py = q.pop()
                count += 1
                if px < minx:
                    minx = px
                if px > maxx:
                    maxx = px
                if py < miny:
                    miny = py
                if py > maxy:
                    maxy = py
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                        rr, gg, bb = pixels[nx, ny]
                        if within(rr, rgb[0]) and within(gg, rgb[1]) and within(bb, rgb[2]):
                            visited[ny][nx] = True
                            q.append((nx, ny))
            cx = (minx + maxx) // 2
            cy = (miny + maxy) // 2
            regions.append({
                "bbox": {"x": minx, "y": miny, "w": maxx - minx + 1, "h": maxy - miny + 1},
                "center": {"x": cx, "y": cy},
                "area": int(count),
            })

    print(json.dumps({
        "event": "list_color_regions",
        "color": hex_color,
        "tolerance": tolerance,
        "direction": direction,
        "count": len(regions),
        "regions": regions,
    }))
    return regions


def snap_to_color(img_path: Path, hex_color: str, monitor_index: int, tolerance: int = 16, direction: str = "lr", next_pixel: bool = False, nearest: bool = False):
    matches = list_buttons_by_color(img_path, hex_color, tolerance=tolerance, direction=direction, nearest=nearest)
    if not matches:
        print(json.dumps({"event": "snap_color", "ok": False, "reason": "no_match"}))
        return False
    idx = 1 if next_pixel and len(matches) > 1 else 0
    px = matches[idx]
    vx, vy, _, _ = get_virtual_screen_rect()
    wx, wy = vx + px["x"], vy + px["y"]
    set_cursor_pos_world(wx, wy)
    time.sleep(0.01)
    cpos = get_cursor_pos()
    ok = (abs(cpos[0] - wx) <= 2 and abs(cpos[1] - wy) <= 2) and IS_WINDOWS
    print(json.dumps({
        "event": "snap_color",
        "ok": bool(ok),
        "color": hex_color,
        "picked": {"x": px["x"], "y": px["y"]},
        "simulated": (not IS_WINDOWS),
    }))
    return ok


def snap_to_color_region(img_path: Path, hex_color: str, monitor_index: int, tolerance: int = 16, direction: str = "lr", next_region: bool = False):
    regions = find_color_regions(img_path, hex_color, tolerance=tolerance, direction=direction)
    if not regions:
        print(json.dumps({"event": "snap_color_region", "ok": False, "reason": "no_region"}))
        return False
    idx = 1 if next_region and len(regions) > 1 else 0
    center = regions[idx]["center"]
    vx, vy, _, _ = get_virtual_screen_rect()
    wx, wy = vx + center["x"], vy + center["y"]
    set_cursor_pos_world(wx, wy)
    time.sleep(0.01)
    cpos = get_cursor_pos()
    ok = (abs(cpos[0] - wx) <= 2 and abs(cpos[1] - wy) <= 2) and IS_WINDOWS
    print(json.dumps({
        "event": "snap_color_region",
        "ok": bool(ok),
        "center": center,
        "region": regions[idx],
        "simulated": (not IS_WINDOWS),
    }))
    return ok


# ------------------------------------------------------------
# Keyboard typing helpers (no-ops on non-Windows)
# ------------------------------------------------------------

def _press_vk(vk: int, shift: bool = False):
    if not IS_WINDOWS:
        return False
    if shift:
        user32.keybd_event(0x10, 0, 0, 0)  # VK_SHIFT down
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    if shift:
        user32.keybd_event(0x10, 0, KEYEVENTF_KEYUP, 0)  # VK_SHIFT up
    return True


def type_text(text: str, press_enter: bool = False):
    performed = IS_WINDOWS
    if IS_WINDOWS:
        for ch in text:
            vk_scan = user32.VkKeyScanW(ord(ch))
            if vk_scan == -1:
                continue
            shift = bool((vk_scan >> 8) & 0x01)
            vk = vk_scan & 0xFF
            _press_vk(vk, shift=shift)
        if press_enter:
            _press_vk(0x0D, shift=False)  # VK_RETURN
    print(json.dumps({"event": "type", "ok": performed, "len": len(text), "enter": bool(press_enter), "simulated": (not IS_WINDOWS)}))
    return performed


# ------------------------------------------------------------
# CLI plumbing (+ self-tests)
# ------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM-friendly CLI for GUI debugging actions (cross-platform safe)")
    sub = p.add_subparsers(dest="cmd")

    # capture
    sp = sub.add_parser("capture", help="Capture virtual desktop to PNG with cursor overlay")
    sp.add_argument("--out", default="current_view.png")
    sp.add_argument("--cursor-type", default="cross", choices=["cross", "dot", "none"])
    sp.add_argument("--cursor-size", type=int, default=15)

    # copy-view
    sp = sub.add_parser("copy-view", help="Copy current_view.png to a work folder")
    sp.add_argument("dest", help="Destination folder for LLM work area")
    sp.add_argument("--src", default="current_view.png")

    # move
    sp = sub.add_parser("move", help="Move mouse to monitor-local coordinates")
    sp.add_argument("--monitor", type=int, required=True)
    sp.add_argument("--x", type=int, required=True)
    sp.add_argument("--y", type=int, required=True)
    sp.add_argument("--recap-out", default=None)

    # click
    sp = sub.add_parser("click", help="Click at monitor-local coordinates")
    sp.add_argument("--monitor", type=int, required=True)
    sp.add_argument("--x", type=int, required=True)
    sp.add_argument("--y", type=int, required=True)
    sp.add_argument("--button", default="left", choices=["left", "right", "middle"])
    sp.add_argument("--double", action="store_true")
    sp.add_argument("--recap-out", default=None)

    # ocr
    sp = sub.add_parser("ocr", help="OCR an image and emit word boxes")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--lang", default="eng")

    # snap-text
    sp = sub.add_parser("snap-text", help="Move cursor to OCR text center (optionally click)")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--query", required=True)
    sp.add_argument("--monitor", type=int, required=False, default=1)
    sp.add_argument("--fuzzy", action="store_true")
    sp.add_argument("--click", action="store_true")

    # list-buttons (raw pixels)
    sp = sub.add_parser("list-buttons", help="List pixels matching color with tolerance")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--color", required=True)
    sp.add_argument("--tol", type=int, default=16)
    sp.add_argument("--direction", default="lr", choices=["lr", "rl", "tb", "bt"])
    sp.add_argument("--nearest", action="store_true")

    # list-color-regions (grouped blobs)
    sp = sub.add_parser("list-color-regions", help="List contiguous color regions (approx buttons)")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--color", required=True)
    sp.add_argument("--tol", type=int, default=16)
    sp.add_argument("--direction", default="lr", choices=["lr", "rl", "tb", "bt"])

    # snap-color (first/next pixel)
    sp = sub.add_parser("snap-color", help="Snap cursor to first/next pixel matching color")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--color", required=True)
    sp.add_argument("--monitor", type=int, required=False, default=1)
    sp.add_argument("--tol", type=int, default=16)
    sp.add_argument("--direction", default="lr", choices=["lr", "rl", "tb", "bt"])
    sp.add_argument("--next-pixel", action="store_true")
    sp.add_argument("--nearest", action="store_true")

    # snap-color-region (first/next region center)
    sp = sub.add_parser("snap-color-region", help="Snap cursor to region center (beyond a single button)")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--color", required=True)
    sp.add_argument("--monitor", type=int, required=False, default=1)
    sp.add_argument("--tol", type=int, default=16)
    sp.add_argument("--direction", default="lr", choices=["lr", "rl", "tb", "bt"])
    sp.add_argument("--next-region", action="store_true")

    # type text (optionally snap to OCR before typing)
    sp = sub.add_parser("type", help="Type text into the current focus (or snap to OCR match then type)")
    sp.add_argument("--text", required=True)
    sp.add_argument("--enter", action="store_true")
    sp.add_argument("--img", default="current_view.png")
    sp.add_argument("--query", default=None, help="OCR text to snap+click before typing")
    sp.add_argument("--monitor", type=int, required=False, default=1)

    # service wrapper (delegates to existing run_cli.py if present)
    sp = sub.add_parser("service", help="Install/Start/Stop/Uninstall capture service via project runner")
    sp.add_argument("--action", required=True, choices=["install", "start", "stop", "uninstall"])
    sp.add_argument("--runner", default="run_cli.py")

    # self-tests (non-interactive verifications without OS hooks)
    sub.add_parser("selftest", help="Run non-interactive self tests and report JSON results")

    return p.parse_args(argv)


def _selftest():
    """Lightweight tests that don't require Windows hooks."""
    results = {
        "platform": sys.platform,
        "is_windows": IS_WINDOWS,
        "mss": bool(mss is not None),
        "pytesseract": bool(pytesseract is not None),
        "steps": []
    }
    # monitors
    mons = enum_monitors()
    results["steps"].append({"name": "enum_monitors", "ok": isinstance(mons, list) and len(mons) >= 1})
    # capture
    try:
        out = Path("current_view.png")
        capture_current_view(out, include_cursor=True)
        ok = out.exists() and out.stat().st_size > 0
        results["steps"].append({"name": "capture_current_view", "ok": ok, "file": str(out)})
    except Exception as e:
        results["steps"].append({"name": "capture_current_view", "ok": False, "error": str(e)})
    # color list on a file (expect ok even if zero results)
    try:
        pixels = list_buttons_by_color(Path("current_view.png"), "#00FF00", tolerance=8)
        results["steps"].append({"name": "list_buttons_by_color", "ok": isinstance(pixels, list)})
    except Exception as e:
        results["steps"].append({"name": "list_buttons_by_color", "ok": False, "error": str(e)})
    print(json.dumps({"event": "selftest", "results": results}))
    return 0 if all(s.get("ok") for s in results["steps"]) else 2


def main(argv=None):
    args = parse_args(argv)

    if args.cmd == "capture":
        include = args.cursor_type != "none"
        capture_current_view(Path(args.out), include_cursor=include, cursor_type=args.cursor_type, cursor_size=args.cursor_size)
        return 0

    if args.cmd == "copy-view":
        copy_current_view_to_folder(Path(args.src), Path(args.dest))
        return 0

    if args.cmd == "move":
        move_mouse_monitor(args.monitor, args.x, args.y, recap_out=Path(args.recap_out) if args.recap_out else None)
        return 0

    if args.cmd == "click":
        click_mouse_monitor(
            args.monitor,
            args.x,
            args.y,
            button=args.button,
            double=args.double,
            recap_out=Path(args.recap_out) if args.recap_out else None,
        )
        return 0

    if args.cmd == "ocr":
        ocr_screen_return_words(Path(args.img), lang=args.lang)
        return 0

    if args.cmd == "snap-text":
        ok = snap_to_text(Path(args.img), args.query, args.monitor, fuzzy=args.fuzzy, click=args.click)
        return 0 if ok else 2

    if args.cmd == "list-buttons":
        list_buttons_by_color(Path(args.img), args.color, tolerance=args.tol, direction=args.direction, nearest=args.nearest)
        return 0

    if args.cmd == "list-color-regions":
        find_color_regions(Path(args.img), args.color, tolerance=args.tol, direction=args.direction)
        return 0

    if args.cmd == "snap-color":
        ok = snap_to_color(
            Path(args.img),
            args.color,
            args.monitor,
            tolerance=args.tol,
            direction=args.direction,
            next_pixel=args.next_pixel,
            nearest=args.nearest,
        )
        return 0 if ok else 2

    if args.cmd == "snap-color-region":
        ok = snap_to_color_region(
            Path(args.img),
            args.color,
            args.monitor,
            tolerance=args.tol,
            direction=args.direction,
            next_region=args.next_region,
        )
        return 0 if ok else 2

    if args.cmd == "type":
        if args.query:
            snap_to_text(Path(args.img), args.query, args.monitor, fuzzy=False, click=IS_WINDOWS)
            time.sleep(0.05)
        type_text(args.text, press_enter=args.enter)
        return 0 if IS_WINDOWS else 2

    if args.cmd == "service":
        # Delegate to project runner if present
        runner = Path(args.runner)
        if not runner.exists():
            print(json.dumps({"event": "service", "ok": False, "error": f"runner not found: {runner}"}))
            return 3
        import subprocess

        cmd = [sys.executable, str(runner), "service", f"--{args.action}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(json.dumps({
                "event": "service",
                "action": args.action,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }))
            return res.returncode
        except Exception as e:
            print(json.dumps({"event": "service", "ok": False, "error": str(e)}))
            return 3

    if args.cmd == "selftest":
        return _selftest()

    # Fallback help
    print(json.dumps({"event": "help", "ok": False}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
