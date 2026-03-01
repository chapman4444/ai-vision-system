# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
pip install -r requirements.txt   # numpy + Pillow only; tkinter/ctypes are built-in

python viewer.pyw                  # Main viewer (tkinter GUI, three-image architecture)
python automation_recorder.pyw     # Floating record/pause/stop panel for step-based automation
python claude_service.py           # Background daemon: continuous capture + file-based IPC
python simple_claude_vision.py capture --message "msg"  # One-shot screenshot to claude_workspace/

START_VISION.bat / STOP_VISION.bat # Start/stop the background daemon
QUICK_CAPTURE.bat                  # Quick one-shot capture
SHOW_CLAUDE.bat                    # Capture with context message for Claude
```

There are no tests or linters configured.

## Three-Image Architecture

The core design solves cursor timing for UI automation. Three files in `claude_session/` serve distinct consumers:

| File | Cursor | Consumer | Purpose |
|------|--------|----------|---------|
| `temp_view.jpg` | No | Viewer display | Base image; viewer composites a live cursor overlay on top |
| `nomouse_cursor.jpg` | No | Change detection | Reference frame; compared against `temp_view.jpg` to detect screen changes without cursor interference |
| `current_view.png` | Baked in | LLM analysis | Cursor position is accurate to the exact capture moment |

The service loop (`claude_service.py` or `viewer.pyw`'s background thread):
1. Capture `temp_view.jpg` (no cursor) every interval (default 2s)
2. Compare against `nomouse_cursor.jpg` using pixel sampling to detect changes
3. If changed, update `nomouse_cursor.jpg` as new reference
4. Capture `current_view.png` with cursor composited at current position
5. Archive to `claude_session/archive/` (max 500 files, 7-day retention)

## Key Modules

**Entry points** (root):
- `viewer.pyw` (~1600 lines) — Main app. Tkinter GUI with minimal command-window chrome, properties dialog, built-in calibration, four window sizing modes (auto_fit/width_based/height_based/fixed with letterboxing). Runs its own capture loop.
- `claude_service.py` — Headless alternative. Runs as background daemon, writes signal files (`READY.txt`/`VIEWED.txt`) and accepts commands via `claude_session/service/commands.json` → `response.json`.
- `viewer_minimal.pyw` — Stripped-down viewer variant.
- `simple_claude_vision.py` — CLI for one-shot captures to `claude_workspace/`.

**`src/capture/`**:
- `gdi_screen_capture.py` — Windows GDI+ capture with double-buffering. Cursor compositing uses PIL ImageDraw, **not** GDI DrawIcon (DrawIcon doesn't work with memory DCs). Supports cross/arrow cursor types.
- `simple_screen_capture.py` — Simplified capture with format selection (PNG/JPEG/WEBP/BMP) and compression levels.

**`src/automation/`**:
- `universal_input.py` — Mouse/keyboard simulation via Win32 API + PyAutoGUI hybrid. Works across browsers, games, native apps. Tracks action history.
- `vision_integration.py` — Wraps capture + input: screenshots before/after actions, validates success, adaptive retry.
- Also: `browser_automation.py`, `game_automation.py`, `gui_elements.py`, `calibration_integration.py`.

**`src/config/`**: `vision_config.py` — Dataclass-based config (CaptureConfig, ServiceConfig, SecurityConfig, StorageConfig) with JSON persistence.

**`src/monitors/`**: `monitor_manager.py` — Multi-monitor enumeration and selection.

**`src/regions/`**: `region_selector.py` — Interactive screen region selection.

**`src/session/`**: `ai_session_manager.py` — Session state and history.

**`src/interface/`**: `claude_interface.py` — Claude integration layer.

**`calibrate/`**: Mouse calibration tools (9-point fullscreen, auto-calibrate, GUI). Needed when screen coordinates don't map 1:1 to cursor positions (e.g., scaling, multi-monitor offsets). Also accessible from viewer's system menu.

## Pitfalls

- **PIL not GDI for cursor drawing**: GDI `DrawIcon` silently fails on memory device contexts. Always use PIL `ImageDraw` for cursor compositing onto captured images.
- **Pixel change detection threshold**: Default is 5% of 100 random sample points with per-pixel color difference > 30. Tunable in `viewer_settings.json` (`pixel_threshold`, `sample_points`, `detection_method`).
- **File-based IPC is polling**: The service checks `commands.json` on each loop iteration. No push notifications. Latency equals capture interval.

## Configuration

All viewer/capture settings persist in `viewer_settings.json` at project root. Key fields: `cursor_type` (cross/cursor/none), `cursor_size`, `sizing_mode`, `capture_interval`, `pixel_threshold`, `detection_method` (random/fixed_coords), `fixed_coords` (semicolon-separated x,y pairs).
